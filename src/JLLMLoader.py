import os
import mmap
import json
import torch
import warnings

class JTensorManager:
    def __init__(self, cache_dir="jweight_cache", window_size=7):
        self.dictTensor = {}     # name -> JTensor (shrunken, CPU)
        self.dictValuePair = {}  # shared global dict of unique pairs
        self.dictValueMulti = {}
        self.gpu_cache = {}      # name -> expanded tensor on GPU (for non-layer tensors)
        self.cache_dir = cache_dir
        self.window_size = window_size  # 1/4 of layers to keep expanded
        self.expanded_layers = set()    # layer indices currently expanded
        self.gpu_weights = {}            # tensorname -> expanded GPU tensor (layer weights)
        self._layer_tensor_prefix = "layers"  # default; overridden by set_layer_tensor_prefix()
        os.makedirs(cache_dir, exist_ok=True)

    def set_layer_tensor_prefix(self, prefix):
        """Set the actual layer tensor prefix (e.g. 'model.layers') used by the model."""
        self._layer_tensor_prefix = prefix

    def _cache_path(self, tensorname):
        safe_name = tensorname.replace("/", "_").replace(".", "_")
        return os.path.join(self.cache_dir, f"{safe_name}.jweight")

    def setTensor(self, tensorname, tensor):
        cache_path = self._cache_path(tensorname)
        bin_path = cache_path + ".bin"
        meta_path = cache_path + ".meta.pt"

        if os.path.exists(bin_path) and os.path.exists(meta_path):
            # Cache exists - load it and store in dictTensor
            j_tensor = JTensor(self.dictValuePair, tensorname, None, cache_path=cache_path)
            j_tensor.load_cache(cache_path)
            self.dictTensor[tensorname] = j_tensor
            return j_tensor

        # RAW storage: just store the tensor directly, no compression
        j_tensor = JTensor(self.dictValuePair, tensorname, tensor, cache_path=cache_path)
        j_tensor.tensor = tensor  # Keep raw tensor, no shrink
        j_tensor.shape = tensor.shape
        j_tensor.original_dtype = tensor.dtype
        j_tensor.save_cache(cache_path)
        self.dictTensor[tensorname] = j_tensor
        return j_tensor

    def getTensor(self, tensorname):
        if tensorname not in self.dictTensor:
            raise KeyError(f"Tensor '{tensorname}' not found.")
        self.dictTensor[tensorname] = self.dictTensor[tensorname].to["cuda"]
        return self.dictTensor[tensorname]

    def get_layer_gpu(self, tensorname, device="cuda"):
        """Get tensor from dictTensor, expand if needed, move to GPU with sliding window management."""
        if tensorname not in self.dictTensor:
            raise KeyError(f"[JTensorManager] Tensor not preloaded: {tensorname}")
        jt = self.dictTensor[tensorname]
        tensor = jt.expandTensor().to(device, non_blocking=True)
        return tensor

    def free_gpu_cache(self):
        """Free all GPU cached tensors."""
        if self.gpu_cache:
            print(f"[JTensorManager] Freeing GPU cache: {len(self.gpu_cache)} tensors")
            for name in list(self.gpu_cache.keys()):
                del self.gpu_cache[name]
            self.gpu_cache.clear()
            torch.cuda.empty_cache()

    def free_gpu_tensor(self, tensorname):
        """Free a specific GPU cached tensor."""
        if tensorname in self.gpu_cache:
            del self.gpu_cache[tensorname]

    def free_cpu_tensor(self, tensorname):
        """Free a specific tensor from CPU dictTensor after it's been moved to GPU."""
        if tensorname in self.dictTensor:
            jt = self.dictTensor[tensorname]
            jt.tensor = None  # Free the tensor data
            jt.locationList = []
            jt.dictValuePair.clear()
            del self.dictTensor[tensorname]

    def free_cpu_layer(self, layer_idx):
        """Free all tensors for a layer from CPU dictTensor."""
        prefix = self._layer_tensor_prefix
        tensor_names = [
            f"layers.{layer_idx}.input_layernorm.weight",
            f"layers.{layer_idx}.post_attention_layernorm.weight",
            f"layers.{layer_idx}.self_attn.q_proj.weight",
            f"layers.{layer_idx}.self_attn.k_proj.weight",
            f"layers.{layer_idx}.self_attn.v_proj.weight",
            f"layers.{layer_idx}.self_attn.o_proj.weight",
            f"layers.{layer_idx}.self_attn.q_proj.bias",
            f"layers.{layer_idx}.self_attn.k_proj.bias",
            f"layers.{layer_idx}.self_attn.v_proj.bias",
            f"layers.{layer_idx}.mlp.gate_proj.weight",
            f"layers.{layer_idx}.mlp.up_proj.weight",
            f"layers.{layer_idx}.mlp.down_proj.weight",
        ]
        for name in tensor_names:
            self.free_cpu_tensor(name)

    def get_layer_weights(self, layer_idx, device="cuda"):
        """Get all weights for a layer, expanding if needed, with sliding window."""
        prefix = self._layer_tensor_prefix
        tensor_names = [
            f"layers.{layer_idx}.input_layernorm.weight",
            f"layers.{layer_idx}.post_attention_layernorm.weight",
            f"layers.{layer_idx}.self_attn.q_proj.weight",
            f"layers.{layer_idx}.self_attn.k_proj.weight",
            f"layers.{layer_idx}.self_attn.v_proj.weight",
            f"layers.{layer_idx}.self_attn.o_proj.weight",
            f"layers.{layer_idx}.self_attn.q_proj.bias",
            f"layers.{layer_idx}.self_attn.k_proj.bias",
            f"layers.{layer_idx}.self_attn.v_proj.bias",
            f"layers.{layer_idx}.mlp.gate_proj.weight",
            f"layers.{layer_idx}.mlp.up_proj.weight",
            f"layers.{layer_idx}.mlp.down_proj.weight",
        ]

        # Manage sliding window - evict old layers if needed
        if len(self.expanded_layers) >= self.window_size:
            oldest = min(self.expanded_layers)
            self._free_layer_gpu_weights(oldest)
            self.expanded_layers.discard(oldest)

        weights = {}
        for name in tensor_names:
            if name not in self.gpu_weights:
                jt = self.dictTensor.get(name)
                if jt is not None:
                    tensor = jt.expandTensor()
                    if tensor is not None:
                        tensor = tensor.to(device, non_blocking=True)
                        self.gpu_weights[name] = tensor
            w = self.gpu_weights.get(name)
            if w is not None:
                weights[name] = w

        self.expanded_layers.add(layer_idx)
        return weights

    def _free_layer_gpu_weights(self, layer_idx):
        """Free all GPU weights for a specific layer."""
        prefix = self._layer_tensor_prefix
        tensor_names = [
            f"layers.{layer_idx}.input_layernorm.weight",
            f"layers.{layer_idx}.post_attention_layernorm.weight",
            f"layers.{layer_idx}.self_attn.q_proj.weight",
            f"layers.{layer_idx}.self_attn.k_proj.weight",
            f"layers.{layer_idx}.self_attn.v_proj.weight",
            f"layers.{layer_idx}.self_attn.o_proj.weight",
            f"layers.{layer_idx}.self_attn.q_proj.bias",
            f"layers.{layer_idx}.self_attn.k_proj.bias",
            f"layers.{layer_idx}.self_attn.v_proj.bias",
            f"layers.{layer_idx}.mlp.gate_proj.weight",
            f"layers.{layer_idx}.mlp.up_proj.weight",
            f"layers.{layer_idx}.mlp.down_proj.weight",
        ]
        for name in tensor_names:
            if name in self.gpu_weights:
                del self.gpu_weights[name]
        if layer_idx in self.expanded_layers:
            self.expanded_layers.remove(layer_idx)

    def _layer_cache_path(self, layer_idx):
        """Get path for combined layer cache file."""
        return os.path.join(self.cache_dir, f"layer_{layer_idx}.jlayer")

    def save_layer_cache(self, layer_idx, loader):
        """Save all tensors for one layer into a single combined file (RAW - no compression)."""
        cache_path = self._layer_cache_path(layer_idx)

        # Collect all tensor names for this layer
        prefix = self._layer_tensor_prefix
        tensor_names = [
            f"layers.{layer_idx}.input_layernorm.weight",
            f"layers.{layer_idx}.post_attention_layernorm.weight",
            f"layers.{layer_idx}.self_attn.q_proj.weight",
            f"layers.{layer_idx}.self_attn.k_proj.weight",
            f"layers.{layer_idx}.self_attn.v_proj.weight",
            f"layers.{layer_idx}.self_attn.o_proj.weight",
            f"layers.{layer_idx}.self_attn.q_proj.bias",
            f"layers.{layer_idx}.self_attn.k_proj.bias",
            f"layers.{layer_idx}.self_attn.v_proj.bias",
            f"layers.{layer_idx}.mlp.gate_proj.weight",
            f"layers.{layer_idx}.mlp.up_proj.weight",
            f"layers.{layer_idx}.mlp.down_proj.weight",
        ]

        layer_data = {}
        for tensorname in tensor_names:
            raw = loader._load_raw_tensor(tensorname)
            if raw is None:
                continue

            layer_data[tensorname] = {
                "shape": raw.shape,
                "dtype": raw.dtype,
                "data": raw.flatten().cpu(),
            }

        bin_path = cache_path + ".bin"
        meta_path = cache_path + ".meta.pt"

        # Write all raw tensors sequentially to binary file
        with open(bin_path, 'wb') as f:
            for tensorname, data in layer_data.items():
                f.write(data["data"].numpy().tobytes())

        # Save metadata
        meta_data = {}
        offset = 0
        for tensorname, data in layer_data.items():
            num_bytes = data["data"].numel() * data["data"].element_size()
            meta_data[tensorname] = {
                "shape": tuple(data["shape"]),
                "dtype": str(data["dtype"]),
                "offset": offset,
                "num_bytes": num_bytes,
            }
            offset += num_bytes

        with open(meta_path, "wb") as f:
            torch.save({
                "layer_idx": layer_idx,
                "tensors": meta_data,
            }, f)

        print(f"[JTensorManager] Saved layer {layer_idx} raw cache: {len(layer_data)} tensors -> {cache_path}")

    def load_layer_cache(self, layer_idx, loader):
        """Load all tensors for one layer from a combined RAW cache file.

        Returns dict of tensorname -> JTensor (tensor already set, no expand needed).
        """
        cache_path = self._layer_cache_path(layer_idx)
        bin_path = cache_path + ".bin"
        meta_path = cache_path + ".meta.pt"

        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Layer cache not found: {cache_path}")

        ck = torch.load(meta_path, map_location="cpu", weights_only=True)

        # Memory-map the binary file
        file_size = os.path.getsize(bin_path)
        storage = torch.UntypedStorage.from_file(bin_path, False, file_size)

        result = {}
        for tensorname, info in ck["tensors"].items():
            offset = info["offset"]
            num_bytes = info["num_bytes"]
            shape = info["shape"]
            dtype_str = info["dtype"]

            if "float16" in dtype_str or "half" in dtype_str:
                dtype = torch.float16
            elif "float32" in dtype_str:
                dtype = torch.float32
            elif "bfloat16" in dtype_str:
                dtype = torch.bfloat16
            else:
                dtype = torch.float16

            # Create tensor view from memory-mapped storage
            tensor = torch.as_tensor(storage[offset:offset + num_bytes], dtype=dtype).clone()
            tensor = tensor.reshape(shape)

            # Wrap in JTensor with tensor already set (no expand needed)
            j_tensor = JTensor({}, tensorname, tensor)
            j_tensor.tensor = tensor
            result[tensorname] = j_tensor

        print(f"[JTensorManager] Loaded layer {layer_idx} cache: {len(result)} tensors (RAW)")
        return result

    def preload_layer(self, layer_idx, loader):
        """Preload all tensors for a layer, using cache if available."""
        cache_path = self._layer_cache_path(layer_idx)

        if not (os.path.exists(cache_path + ".bin") and os.path.exists(cache_path + ".meta.pt")):
            # Save layer cache for future use if it doesn't exist
            self.save_layer_cache(layer_idx, loader)

        jtensors = self.load_layer_cache(layer_idx, loader)
        for tensorname, j_tensor in jtensors.items():
            self.dictTensor[tensorname] = j_tensor

    def save_all_layers(self, loader, num_layers):
        """Save all layers to individual layer cache files.

        Args:
            loader: JLLMLoader instance
            num_layers: Total number of layers to save
        """
        for layer_idx in range(num_layers):
            cache_path = self._layer_cache_path(layer_idx)
            bin_path = cache_path + ".bin"
            meta_path = cache_path + ".meta.pt"

            if os.path.exists(bin_path) and os.path.exists(meta_path):
                print(f"[JTensorManager] Layer {layer_idx} cache already exists, skipping.")
                continue

            print(f"[JTensorManager] Saving layer {layer_idx}/{num_layers-1}...")
            self.save_layer_cache(layer_idx, loader)

        print(f"[JTensorManager] All {num_layers} layers saved.")

import os
import mmap
import torch

class JTensor:
    def __init__(self, dictValuePair, tensorName, tensor, cache_path=None):
        self.dictValuePair = dictValuePair  # pair keys → [v0, v1] pairs
        self.locationGroups = {}             # location group keys → [v0, v1] groups
        self.locationList = []              # tracking loop steps for reconstruction
        self.tensorName = tensorName
        self.tensor = tensor
        self.shape = tensor.shape if tensor is not None else None
        self.original_dtype = tensor.dtype if tensor is not None else None
        self.unique_keys_tracked = []       # unique pair index keys
        self.cache_path = cache_path        # optional pre-existing cache path
        self._cache_file_handle = None
        self._cache_mmap_obj = None

    def save_cache(self, cache_path_base):
        bin_path = cache_path_base + ".bin"
        meta_path = cache_path_base + ".meta.pt"

        if self.tensor is not None:
            # RAW storage: save the raw tensor directly
            raw_tensor = self.tensor.flatten().cpu()
            num_elements = raw_tensor.numel()
            raw_np = raw_tensor.numpy()
            with open(bin_path, 'wb') as f:
                f.write(raw_np.tobytes())

            with open(meta_path, "wb") as f:
                torch.save({
                    "shape": self.shape,
                    "dtype": self.original_dtype,
                    "is_raw": True,
                }, f)
        else:
            # Compressed storage: save packed indices
            packed_tensor = self.locationList[0].cpu().contiguous()
            num_elements = packed_tensor.numel()
            packed_np = packed_tensor.numpy()
            with open(bin_path, 'wb') as f:
                f.write(packed_np.tobytes())

            keys_to_save = self.unique_keys_tracked
            if isinstance(keys_to_save, torch.Tensor):
                keys_to_save = keys_to_save.cpu()
            elif isinstance(keys_to_save, list):
                keys_to_save = torch.tensor(keys_to_save)

            # FIX: Consolidated the duplicate meta saves into one complete dictionary
            with open(meta_path, "wb") as f:
                torch.save({
                    "shape": self.shape,
                    "dtype": self.original_dtype,
                    "is_raw": False,
                    "packed_dtype": packed_tensor.dtype,
                    "num_elements": num_elements,
                    "unique_keys_tracked": keys_to_save,
                    "dictValuePair": self.dictValuePair,
                    "locationGroups": self.locationGroups,
                }, f)
            
        print(f"[JTensor] Saved mmap cache files: {bin_path} & {meta_path}")

    def load_cache(self, cache_path_base):
        bin_path = cache_path_base + ".bin"
        meta_path = cache_path_base + ".meta.pt"

        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing cache files for {cache_path_base}")

        # Clean up existing file handle before mapping
        if self._cache_file_handle is not None:
            try:
                self._cache_mmap_obj.close()
                self._cache_file_handle.close()
            except Exception:
                pass

        ck = torch.load(meta_path, map_location="cpu", weights_only=True)
        self.shape = torch.Size(ck["shape"])
        self.original_dtype = ck["dtype"]

        if ck.get("is_raw", False):
            # RAW storage: load tensor directly via memory-map
            file_size = os.path.getsize(bin_path)
            
            # FIX: Read the binary file directly into a buffer, interpret with correct dtype, then clone
            # This avoids UntypedStorage byte-slicing mismatch errors
            with open(bin_path, "rb") as f:
                byte_data = bytearray(f.read())
                
            raw_tensor = torch.frombuffer(byte_data, dtype=self.original_dtype).clone()
            
            # Cast to float32 as originally intended and reshape
            self.tensor = raw_tensor.to(torch.float32).reshape(self.shape)
            self.locationList = []
            self._cache_file_handle = None
            self._cache_mmap_obj = None
            print(f"[JTensor] Loaded RAW cache: {bin_path}")
            
        else:
            # Compressed storage: load packed indices
            raw_keys = ck["unique_keys_tracked"]
            if isinstance(raw_keys, torch.Tensor):
                self.unique_keys_tracked = raw_keys.tolist()
            else:
                self.unique_keys_tracked = list(raw_keys)

            self.dictValuePair.clear()
            for k, v in ck["dictValuePair"].items():
                self.dictValuePair[int(k)] = v

            self.locationGroups = ck.get("locationGroups", {})

            packed_dtype = ck["packed_dtype"]
            num_elements = ck["num_elements"]

            self._cache_file_handle = open(bin_path, "rb")
            self._cache_mmap_obj = mmap.mmap(
                self._cache_file_handle.fileno(),
                0,
                access=mmap.ACCESS_READ
            )

            mmapped_tensor = torch.frombuffer(
                self._cache_mmap_obj,
                dtype=packed_dtype,
                count=num_elements
            )

            self.locationList = [mmapped_tensor]
            self.tensor = None
            print(f"[JTensor] Loaded compressed cache: {bin_path}")
    
    def unique(self, pairs):
        scale_factor = 1000.0

        if pairs.dtype == torch.float32:
            scaled = (pairs * scale_factor).to(torch.int32)
        elif pairs.dtype == torch.float16:
            scaled = (pairs * scale_factor).to(torch.int16).to(torch.int32)
        else:
            raise TypeError("Input must be float16 or float32")

        m = 65536

        # Pack pairs: keys[i] = scaled[i*2] * m + scaled[i*2+1]
        num_pairs = pairs.size(0)
        pair_keys = scaled[:, 0] * m + scaled[:, 1]

        unique_keys, inverse_indices = torch.unique(pair_keys, return_inverse=True)
        num_unique = unique_keys.size(0)

        # Find first occurrence of each unique key
        first_occurrence = torch.zeros(num_unique, dtype=torch.long, device=pairs.device)
        first_occurrence.scatter_(
            0,
            inverse_indices.flip(0),
            torch.arange(num_pairs - 1, -1, -1, device=pairs.device)
        )

        # Get the actual pairs for each unique key
        unique_pairs = pairs[first_occurrence]
        return unique_pairs, inverse_indices

    def shrinkTensor(self):
        """Shrink: dynamic quadruplet mapping using row uniqueness and variable dtype fitting."""
        if self.tensor is None:
            return self

        if self.cache_path is not None:
            bin_path = self.cache_path + ".bin"
            meta_path = self.cache_path + ".meta.pt"
            if os.path.exists(bin_path) and os.path.exists(meta_path):
                print(f"[JTensor] ✓ Cache found at {self.cache_path}, loading instead of compressing")
                self.load_cache(self.cache_path)
                return self

        t = self.tensor.flatten()
        
        remainder = t.shape[0] % 16
        if remainder != 0:
            t = t[:-remainder]
            self.shape = torch.Size([t.shape[0]])
        else:
            self.shape = self.tensor.shape
            
        self.original_dtype = self.tensor.dtype

        quads = t.view(-1, 16)
        unique_quads, quad_inv = self.unique(quads)
        
        num_unique = unique_quads.size(0)

        if num_unique == 300:
            raise ValueError(
                f"Compression failed: Found {num_unique} unique items. "
                f"torch.uint8 can only map a maximum of 256 distinct values safely without losing data accuracy."
            )

        if num_unique <= 256:
            optimized_dtype = torch.uint8
        elif num_unique <= 32768:
            optimized_dtype = torch.int16
        else:
            optimized_dtype = torch.int32
            
        quad_inv_packed = quad_inv.to(optimized_dtype)
        
        self.unique_keys_tracked = list(range(num_unique))
        self.dictValuePair.clear()
        unique_np = unique_quads.cpu().numpy()
        for i in range(num_unique):
            self.dictValuePair[i] = unique_np[i].tolist()

        self.locationList = [quad_inv_packed.cpu()]
        self.tensor = None
        return self
        
    def expandTensor(self):
        if self.tensor is not None:
            return self.tensor
        if not self.locationList:
            return None

        quad_inv = self.locationList[0].to(torch.long)
        device = quad_inv.device
        
        unique_values_list = [self.dictValuePair[i] for i in self.unique_keys_tracked]
        unique_quads = torch.tensor(unique_values_list, dtype=self.original_dtype, device=device)

        expanded = unique_quads[quad_inv].flatten()

        expected_len = self.shape.numel()
        current_len = expanded.shape[0]
        
        if current_len < expected_len:
            padding = torch.zeros(expected_len - current_len, dtype=self.original_dtype, device=device)
            expanded = torch.cat([expanded, padding])
        elif current_len > expected_len:
            expanded = expanded[:expected_len]

        self.tensor = expanded.view(self.shape)
        return self.tensor
    

class JLLMLoader:
    """
    Memory-mapped loader for .jllm format.
    [1MB Header (JSON)] [Binary Tensor Data]
    """
    DTYPE_MAP = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    def __init__(self, jllm_filepath, tokenizer=None):
        self.tensorManager = JTensorManager()
        self.filepath = jllm_filepath
        self.file_handle = open(jllm_filepath, "rb")
        self.file_handle.seek(0)
        header_bytes = self.file_handle.read(1024 * 1024).rstrip(b'\x00')
        self.header = json.loads(header_bytes.decode('utf-8'))
        self.mmap_obj = mmap.mmap(self.file_handle.fileno(), 0, access=mmap.ACCESS_READ)
        self.tokenizer = tokenizer
        self.tensor_cache = {}
        num_tensors = len(self.header.get("tensors", {}))
        print(f"[Vault Active] Loading cache: {jllm_filepath}")
        print(f"[Vault Active] {num_tensors} matrices loaded via mmap.")

        # Detect actual layer tensor prefix and propagate to tensorManager
        layer_prefix = self._detect_layer_tensor_prefix()
        self.tensorManager.set_layer_tensor_prefix(layer_prefix)
        print(f"[Vault Active] Layer tensor prefix: '{layer_prefix}'")

    def _detect_layer_tensor_prefix(self):
        """Detect the actual prefix for layer tensors (e.g. 'model.layers' vs 'layers')."""
        all_tensors = list(self.header.get("tensors", {}).keys())
        # Match patterns like "model.layers.0.self_attn.q_proj.weight" or "layers.0...."
        import re
        for tensor in all_tensors:
            # Try to find "something.layers.<digit>."
            m = re.search(r'(\S+\.layers)\.\d+\.', tensor)
            if m:
                return m.group(1)
        # Fallback: check for bare "layers.<digit>."
        for tensor in all_tensors:
            m = re.search(r'(layers)\.\d+\.', tensor)
            if m:
                return m.group(1)
        return "layers"  # default fallback

    def _load_raw_tensor(self, tensor_name: str):
        if tensor_name not in self.header.get("tensors", {}):
            return None
        meta = self.header["tensors"][tensor_name]
        offset = meta["offset"]
        shape = meta["shape"]
        dtype_str = meta.get("dtype", "float16")
        pt_dtype = self.DTYPE_MAP.get(dtype_str, torch.float16)
        
        numel = 1
        for dim in shape:
            numel *= dim
            
        with warnings.catch_warnings():
            tensor = torch.frombuffer(self.mmap_obj, dtype=pt_dtype, count=numel, offset=offset)
        return tensor.reshape(shape)

    def _extract_layer_idx(self, tensor_name):
        """Extract layer index from tensor name like 'model.layers.0.self_attn.q_proj.weight'."""
        import re
        # Match <prefix>.layers.<digit>. or just layers.<digit>.
        m = re.search(r'(\S+)\.layers\.(\d+)\.', tensor_name)
        if m:
            return int(m.group(2))
        return None

    def get_matrix(self, tensor_name: str, target_device: str = "cuda"):
        if tensor_name not in self.header.get("tensors", {}):
            return None
        if tensor_name not in self.tensorManager.dictTensor:
            # Check if this tensor belongs to a layer that can be loaded from layer cache
            layer_idx = self._extract_layer_idx(tensor_name)
            if layer_idx is None:
                raw = self._load_raw_tensor(tensor_name)
                self.tensorManager.setTensor(tensor_name, raw)

        jt = self.tensorManager.dictTensor[tensor_name]
        tensor = jt.expandTensor() if jt.tensor is None else jt.tensor.to("cuda")

        if target_device.startswith("cuda"):
            tensor = tensor.to(target_device, non_blocking=True)
            # Cache the GPU tensor to avoid repeated transfers
            self.tensorManager.gpu_cache[tensor_name] = tensor
        elif target_device == "cpu":
            tensor = tensor.cpu()
        return tensor

    def preload_all_layers(self, verbose=True):
        all_names = self.list_tensors()
        print(f"\n[JLLMLoader] Preloading {len(all_names)} tensors into RAM...")
        for name in all_names:
            raw = self._load_raw_tensor(name)
            if raw is None:
                continue
            self.tensorManager.setTensor(name, raw)
            if verbose:
                jt = self.tensorManager.dictTensor.get(name)
                pairs_info = len(jt.dictValuePair) if jt else 0
                print(f"  [RAM] {name} -> shape {raw.shape}, unique signatures: {pairs_info}")
        print(f"[JLLMLoader] Preload complete. {len(self.tensorManager.dictTensor)} tensors in RAM.")

    def get_layer_gpu(self, tensorname, device="cuda"):
        return self.tensorManager.get_layer_gpu(tensorname, device)

    def get_layer_cpu(self, tensorname):
        jt = self.tensorManager.dictTensor.get(tensorname)
        if jt is None:
            raise KeyError(f"[JLLMLoader] Tensor not preloaded: {tensorname}")
        jt.expandTensor()
        return jt.tensor

    def get_architecture(self):
        return self.header.get("architecture", {})

    def get_model_type(self):
        return self.header.get("model_type", "Unknown")

    def list_tensors(self):
        return list(self.header.get("tensors", {}).keys())

    def close(self):
        if hasattr(self, 'mmap_obj'):
            try:
                self.mmap_obj.close()
            except Exception:
                pass
        if hasattr(self, 'file_handle'):
            try:
                self.file_handle.close()
            except Exception:
                pass
        print("[Vault Closed] Weight store released.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

def read_jllm_header(filepath):
    with open(filepath, "rb") as f:
        header_bytes = f.read(1024 * 1024).rstrip(b'\x00')
        return json.loads(header_bytes.decode('utf-8'))
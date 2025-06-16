import torch
import math

class FrequencyBandSplit:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "low_freq_end": ("FLOAT", {"default": 0.15, "min": 0.05, "max": 0.5, "step": 0.01}),
                "mid_freq_end": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 0.8, "step": 0.01}),
                "overlap": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.3, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "LATENT", "LATENT")
    RETURN_NAMES = ("low_freq", "mid_freq", "high_freq")
    FUNCTION = "split_frequency"
    CATEGORY = "latent/frequency"
    
    def split_frequency(self, latent, low_freq_end, mid_freq_end, overlap):
        samples = latent["samples"]
        
        # Convert to frequency domain
        freq_domain = torch.fft.fftn(samples, dim=(-2, -1))
        
        # Create masks for each band
        h, w = samples.shape[-2:]
        masks = self.create_frequency_masks(h, w, low_freq_end, mid_freq_end, overlap)
        
        bands = []
        for mask in masks:
            mask = mask.to(samples.device).unsqueeze(0).unsqueeze(0)
            band_freq = freq_domain * mask
            band_latent = torch.fft.ifftn(band_freq, dim=(-2, -1)).real
            bands.append({"samples": band_latent})
        
        return tuple(bands)
    
    def create_frequency_masks(self, h, w, low_end, mid_end, overlap):
        center_h, center_w = h // 2, w // 2
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y, x = y.float(), x.float()
        
        # Distance from center
        y_centered = y - center_h
        x_centered = x - center_w
        max_dist = math.sqrt(center_h**2 + center_w**2)
        dist_norm = torch.sqrt(x_centered**2 + y_centered**2) / max_dist
        
        # Create three band masks
        low_mask = torch.sigmoid((low_end - dist_norm) / overlap * 10)
        mid_mask = torch.sigmoid((dist_norm - low_end + overlap) / overlap * 10) * torch.sigmoid((mid_end - dist_norm) / overlap * 10)
        high_mask = torch.sigmoid((dist_norm - mid_end + overlap) / overlap * 10)
        
        return [low_mask, mid_mask, high_mask]

class FrequencyBandCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "low_freq": ("LATENT",),
                "mid_freq": ("LATENT",),
                "high_freq": ("LATENT",),
                "method": (["frequency_domain", "weighted_blend"], {"default": "weighted_blend"}),
                "low_weight": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1}),
                "mid_weight": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),
                "high_weight": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "combine_frequency"
    CATEGORY = "latent/frequency"
    
    def combine_frequency(self, low_freq, mid_freq, high_freq, method, low_weight, mid_weight, high_weight):
        if method == "weighted_blend":
            # Simple weighted combination
            total_weight = low_weight + mid_weight + high_weight
            if total_weight > 0:
                combined = (low_freq["samples"] * low_weight + 
                           mid_freq["samples"] * mid_weight + 
                           high_freq["samples"] * high_weight) / total_weight
            else:
                combined = (low_freq["samples"] + mid_freq["samples"] + high_freq["samples"]) / 3
        else:
            # Frequency domain reconstruction
            combined = low_freq["samples"] + mid_freq["samples"] + high_freq["samples"]
        
        return ({"samples": combined},)

NODE_CLASS_MAPPINGS = {
    "FrequencyBandSplit": FrequencyBandSplit,
    "FrequencyBandCombine": FrequencyBandCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrequencyBandSplit": "Jurdns Frequency Split",
    "FrequencyBandCombine": "Jurdns Frequency Combine",
}
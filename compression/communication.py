import torch


def calculate_state_dict_size_mb(state_dict: dict) -> float:
    """
    Calculates the dense size of a model state dictionary in MB.
    """
    total_bytes = 0

    for key, tensor in state_dict.items():
        total_bytes += tensor.numel() * tensor.element_size()

    return total_bytes / (1024.0 * 1024.0)


def calculate_actual_sparse_size_mb(
    state_dict: dict,
    k_ratio: float = 0.20
) -> float:
    """
    Calculates the estimated transmitted size of a Top-K sparse update.

    For sparsified floating-point tensors:
        value = 4 bytes
        index = 4 bytes

    Normalization buffers are kept dense.
    """

    from compression.top_k import _NEVER_SPARSIFY_SUBSTRINGS

    total_bytes = 0

    for key, tensor in state_dict.items():

        if tensor.is_floating_point():

            # Normalization statistics remain dense
            if any(s in key for s in _NEVER_SPARSIFY_SUBSTRINGS):

                total_bytes += tensor.numel() * tensor.element_size()

            else:

                non_zeros = max(
                    1,
                    int(tensor.numel() * k_ratio)
                )

                # Value + index
                total_bytes += non_zeros * (
                    tensor.element_size() + 4
                )

        else:
            total_bytes += tensor.numel() * tensor.element_size()

    return total_bytes / (1024.0 * 1024.0)


class CommunicationTracker:

    def __init__(
        self,
        num_hospitals: int = 4,
        uncompressed_size_mb: float = 48.0
    ):

        self.num_hospitals = num_hospitals
        self.uncompressed_size_mb = uncompressed_size_mb

        self.total_uplink_uncompressed_mb = 0.0
        self.total_uplink_compressed_mb = 0.0

        self.total_roundtrip_uncompressed_mb = 0.0
        self.total_roundtrip_compressed_mb = 0.0

    def log_round(
        self,
        global_weights: dict,
        k_ratio: float = 0.20,
        enabled: bool = True
    ):

        # Server -> Hospital
        # This remains dense.
        download_size = calculate_state_dict_size_mb(
            global_weights
        )

        # Hospital -> Server
        # This is where Top-K compression is applied.
        if enabled:

            upload_size = calculate_actual_sparse_size_mb(
                global_weights,
                k_ratio=k_ratio
            )

        else:

            upload_size = download_size

        # -------------------------------
        # UPLINK COMMUNICATION
        # -------------------------------

        uplink_uncompressed = (
            self.num_hospitals * download_size
        )

        uplink_compressed = (
            self.num_hospitals * upload_size
        )

        self.total_uplink_uncompressed_mb += (
            uplink_uncompressed
        )

        self.total_uplink_compressed_mb += (
            uplink_compressed
        )

        # -------------------------------
        # ROUND-TRIP COMMUNICATION
        # -------------------------------

        roundtrip_uncompressed = (
            self.num_hospitals
            * (download_size + download_size)
        )

        roundtrip_compressed = (
            self.num_hospitals
            * (download_size + upload_size)
        )

        self.total_roundtrip_uncompressed_mb += (
            roundtrip_uncompressed
        )

        self.total_roundtrip_compressed_mb += (
            roundtrip_compressed
        )

    def get_summary(self) -> dict:

        # Uplink reduction
        if self.total_uplink_uncompressed_mb > 0:

            uplink_reduction = (
                1.0
                - (
                    self.total_uplink_compressed_mb
                    / self.total_uplink_uncompressed_mb
                )
            ) * 100.0

        else:

            uplink_reduction = 0.0

        # Overall round-trip reduction
        if self.total_roundtrip_uncompressed_mb > 0:

            roundtrip_reduction = (
                1.0
                - (
                    self.total_roundtrip_compressed_mb
                    / self.total_roundtrip_uncompressed_mb
                )
            ) * 100.0

        else:

            roundtrip_reduction = 0.0

        return {
            "uncompressed_total_mb": round(self.total_roundtrip_uncompressed_mb, 2),
            "compressed_total_mb": round(self.total_roundtrip_compressed_mb, 2),
            "communication_reduction_percent": round(roundtrip_reduction, 1),
            "uplink_uncompressed_mb": round(self.total_uplink_uncompressed_mb, 2),
            "uplink_compressed_mb": round(self.total_uplink_compressed_mb, 2),
            "uplink_reduction_percent": round(uplink_reduction, 1),
            "roundtrip_uncompressed_mb": round(self.total_roundtrip_uncompressed_mb, 2),
            "roundtrip_compressed_mb": round(self.total_roundtrip_compressed_mb, 2),
            "roundtrip_reduction_percent": round(roundtrip_reduction, 1)
        }
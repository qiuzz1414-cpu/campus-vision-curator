"""六个流水线阶段，每个阶段只读写 manifest，可独立运行、可断点续跑。"""

STAGES = ("ingest", "dedup", "label", "visualize", "export", "train")

__all__ = ["STAGES"]

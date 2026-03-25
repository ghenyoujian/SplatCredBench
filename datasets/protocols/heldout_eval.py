def build_heldout_protocol(num_views: int=10) -> dict[str, list[int]]:
    return {"train": list(range(max(0,num_views-2))), "heldout": list(range(max(0,num_views-2), num_views))}

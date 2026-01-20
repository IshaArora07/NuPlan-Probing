if hasattr(feat, "is_valid") and (not feat.is_valid):
            # try to expose why
            keys = list(feat.data.keys()) if hasattr(feat, "data") else []
            ref_ok = None
            ref_lines = None
            if "reference_line" in feat.data:
                vm = feat.data["reference_line"]["valid_mask"]
                ref_ok = bool(vm.any()) if vm is not None else False
                ref_lines = int(vm.shape[0]) if hasattr(vm, "shape") else None

            map_polys = None
            if "map" in feat.data and "point_position" in feat.data["map"]:
                map_polys = int(feat.data["map"]["point_position"].shape[0])

            raise RuntimeError(
                "[PlutoFeatureBuilder] Produced INVALID feature (would cause compute_features=None)\n"
                f"  token={scenario_token}\n"
                f"  feat.keys={keys}\n"
                f"  reference_line_present={'reference_line' in feat.data}\n"
                f"  reference_line_any_valid={ref_ok}, num_ref_lines={ref_lines}\n"
                f"  map_num_polygons={map_polys}\n"
                f"  route_roadblock_ids_len={len(scenario.get_route_roadblock_ids())}\n"
                f"  map_api_is_None={scenario.map_api is None}\n"
            )

        return feat

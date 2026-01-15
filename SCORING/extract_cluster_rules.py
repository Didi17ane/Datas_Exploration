import json
from pathlib import Path

SRC = Path("/home/didi/jnotebook/Datas_Exploration/SCORING/Rules Clusters/cluster_rules.json")

def load_rules(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # support dict {cluster: rules} or list of records [{"cluster":.., "content":..}, ...] or [{"cluster":.., "rules":..},...]
    rules_map = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                rules_map[int(k)] = v
            except Exception:
                rules_map[k] = v
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                if "cluster" in item:
                    key = item["cluster"]
                    # possible key types
                    try:
                        key = int(key)
                    except Exception:
                        pass
                    if "rules" in item:
                        rules_map[key] = item["rules"]
                    elif "content" in item:
                        rules_map[key] = item["content"]
                    else:
                        # if item itself is the rules list and cluster omitted, skip
                        pass
    return rules_map

def pretty_print_rules(rules_map):
    first = True
    for cluster in sorted(rules_map.keys(), key=lambda x: int(x) if isinstance(x, (int,str)) and str(x).isdigit() else x):
        rules = rules_map[cluster]
        print(f"{cluster}: [")
        if isinstance(rules, list):
            for cond in rules:
                # cond expected as dict like {"sex": ["Féminin"]}
                cond_json = json.dumps(cond, ensure_ascii=False)
                print(f"            {cond_json},")
        else:
            # fallback: dump whole element
            print(f"            {json.dumps(rules, ensure_ascii=False)},")
        print("        ],")
    # optional trailing newline
    print()

if __name__ == "__main__":
    rules_map = load_rules(SRC)
    pretty_print_rules(rules_map)
"""
对缺失字段（formation_energy_per_atom / energy_above_hull / cbm / vbm / is_stable）
       用summary端点做批量查询（一次请求带一批material_ids，而不是像DOS那样逐个请求，
       summary端点是支持批量的，DOS之所以逐个请求是因为get_dos_from_material_id本身
       只能单个查）
把结果写成 {mp_id}_metadata.json（更新/追加）+ 一份汇总 real_labels.csv


使用前必须做的事（这个环境没有网络也没装mp_api，没法帮你实测）:
    1. `pip install mp-api`
    2. 先对1个mp_id单独跑一下 debug_single_query()，核对返回字段名和你预期的一致，
       尤其是cbm/vbm——不少mp_api版本里这两个字段不在summary端点，
       需要换electronic_structure端点查，debug_single_query()里有相应提示。
    3. 确认无误后再跑 batch_fetch_real_labels() 批量执行。
"""

import os
import glob
import json
import time
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
# ============================================================
# 第一部分：从CIF目录解析mp_id（和你两个脚本用一致的逻辑）
# ============================================================

def list_mp_ids(cif_dir: str = "data/cif_files") -> list:
    cif_files = glob.glob(os.path.join(cif_dir, "*.cif"))
    mp_ids = []
    for cif_file in cif_files:
        filename = os.path.basename(cif_file)
        if filename.startswith("mp-") and filename.endswith(".cif"):
            mp_ids.append(filename.replace(".cif", ""))
    return sorted(mp_ids)


# ============================================================
# 第二部分：单个mp_id先手动核对一遍字段（强烈建议先跑这个）
# ============================================================

def debug_single_query(api_key: str, mp_id: str):
    """
    用一个mp_id把summary端点能返回的字段全部打印出来，
    你人工核对一下 formation_energy_per_atom / energy_above_hull /
    cbm / vbm / is_stable 这几个字段名和结构是否符合预期，
    再决定要不要改下面 REQUESTED_FIELDS 里的字段名。
    """
    from mp_api.client import MPRester

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            material_ids=[mp_id],
            fields=[
                "material_id", "formula_pretty", "band_gap",
                "formation_energy_per_atom", "energy_above_hull",
                "is_stable", "cbm", "vbm", "efermi",
            ],
        )
        if not docs:
            print(f"没查到 {mp_id}")
            return
        doc = docs[0]
        print(f"--- {mp_id} 原始返回字段 ---")
        for field in [
            "material_id", "formula_pretty", "band_gap",
            "formation_energy_per_atom", "energy_above_hull",
            "is_stable", "cbm", "vbm", "efermi",
        ]:
            print(f"  {field}: {getattr(doc, field, '<字段不存在>')!r}")

        if getattr(doc, "cbm", None) is None and getattr(doc, "vbm", None) is None:
            print(
                "\n⚠️ summary端点里没查到cbm/vbm（不少mp_api版本这两个字段"
                "不在summary里，而是在electronic_structure/bandstructure端点下）。"
                "如果确实没有，试试："
                "mpr.materials.electronic_structure.search(material_ids=[mp_id], "
                "fields=['cbm', 'vbm', 'band_gap'])，"
                "字段名以你实际拿到的为准，再回来改REQUESTED_FIELDS和下面的解析逻辑。"
            )


# ============================================================
# 第三部分：批量补充真实标签
# ============================================================

REQUESTED_FIELDS = [
    "material_id",
    "formula_pretty",
    "band_gap",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "cbm",
    "vbm",
    "efermi",
]


def batch_fetch_real_labels(
    api_key: str,
    cif_dir: str = "data/cif_files",
    dos_dir: str = "data/dos_data",
    output_csv: str = "data/real_labels.csv",
    chunk_size: int = 500,
    reuse_existing_metadata: bool = True,
) -> pd.DataFrame:
    """
    批量拉取真实标签。summary端点支持一次传一批material_ids，
    所以这里按chunk_size分批请求，而不是像DOS那样逐个请求，
    能显著减少请求次数。
    """
    from mp_api.client import MPRester

    mp_ids = list_mp_ids(cif_dir)
    print(f"共 {len(mp_ids)} 个mp_id待处理")

    # 先看看有多少能从已有metadata.json里白捡band_gap，减少重复请求的必要性
    # （即便如此，为了拿formation_energy等新字段，summary请求还是要发，
    # 这里只是提示复用，不会真的跳过请求）
    if reuse_existing_metadata:
        n_have_meta = sum(
            os.path.exists(os.path.join(dos_dir, f"{mid}_metadata.json"))
            for mid in mp_ids
        )
        print(f"其中 {n_have_meta} 个已有dos metadata.json（band_gap可复用，"
              f"但formation_energy等新字段仍需请求）")

    records = []
    with MPRester(api_key) as mpr:
        for i in range(0, len(mp_ids), chunk_size):
            chunk = mp_ids[i:i + chunk_size]
            print(f"请求第 {i // chunk_size + 1} 批，共 {len(chunk)} 个material_id ...")
            try:
                docs = mpr.materials.summary.search(
                    material_ids=chunk,
                    fields=REQUESTED_FIELDS,
                )
            except Exception as e:
                print(f"  ✗ 这一批请求失败：{e}，改为逐个重试")
                docs = []
                for mid in chunk:
                    try:
                        d = mpr.materials.summary.search(
                            material_ids=[mid], fields=REQUESTED_FIELDS
                        )
                        docs.extend(d)
                    except Exception as e2:
                        print(f"    ✗ {mid} 仍然失败：{e2}")
                    time.sleep(0.1)

            found_ids = set()
            for doc in docs:
                mid = str(doc.material_id)
                found_ids.add(mid)

                cbm = getattr(doc, "cbm", None)
                vbm = getattr(doc, "vbm", None)

                record = {
                    "mp_id": mid,
                    "formula": getattr(doc, "formula_pretty", None),
                    "band_gap_mp": getattr(doc, "band_gap", None),
                    "formation_energy_per_atom_mp": getattr(
                        doc, "formation_energy_per_atom", None
                    ),
                    "energy_above_hull_mp": getattr(doc, "energy_above_hull", None),
                    "is_stable_mp": getattr(doc, "is_stable", None),
                    "cbm_mp": cbm,
                    "vbm_mp": vbm,
                    "efermi_mp": getattr(doc, "efermi", None),
                    "has_band_edges_mp": (cbm is not None) and (vbm is not None),
                }
                records.append(record)

                # 顺手把补充字段写回/更新 {mp_id}_metadata.json，
                # 和 get_dos_data_from_mp.py 的产出对齐、合并成一份
                meta_path = os.path.join(dos_dir, f"{mid}_metadata.json")
                existing_meta = {}
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        existing_meta = json.load(f)
                existing_meta.update(record)
                with open(meta_path, "w") as f:
                    json.dump(existing_meta, f, indent=2)

            missing = set(chunk) - found_ids
            if missing:
                print(f"  ⚠ 这一批里 {len(missing)} 个material_id没查到summary数据: "
                      f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
                for mid in missing:
                    records.append({
                        "mp_id": mid, "formula": None, "band_gap_mp": None,
                        "formation_energy_per_atom_mp": None,
                        "energy_above_hull_mp": None, "is_stable_mp": None,
                        "cbm_mp": None, "vbm_mp": None, "efermi_mp": None,
                        "has_band_edges_mp": False,
                    })

            time.sleep(0.2)

    df = pd.DataFrame(records).drop_duplicates(subset="mp_id").set_index("mp_id")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv)

    n_bg = df["band_gap_mp"].notna().sum()
    n_fe = df["formation_energy_per_atom_mp"].notna().sum()
    n_be = df["has_band_edges_mp"].sum()
    print(f"\n完成。共 {len(df)} 条：")
    print(f"  band_gap 覆盖率: {n_bg}/{len(df)} ({n_bg/len(df)*100:.1f}%)")
    print(f"  formation_energy_per_atom 覆盖率: {n_fe}/{len(df)} ({n_fe/len(df)*100:.1f}%)")
    print(f"  cbm/vbm 覆盖率: {n_be}/{len(df)} ({n_be/len(df)*100:.1f}%) "
          f"—— 如果这里是0，大概率summary端点不带这两个字段，去看"
          f"debug_single_query()打印的提示，改用electronic_structure端点查")
    print(f"结果已存到 {output_csv}，并同步更新了 {dos_dir} 下的 metadata.json")
    return df


# ============================================================
if __name__ == "__main__":
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError(
            "请先设置环境变量 MP_API_KEY，不要把key写死在脚本里。"
            "例如：export MP_API_KEY=你的key"
        )

    # 第一步：强烈建议先跑一个单点debug，核对字段名
    mp_ids = list_mp_ids("data/cif_files")
    if mp_ids:
        debug_single_query(api_key, mp_ids[0])
        input("\n核对完上面的字段无误后，按回车继续批量请求...")

    # 第二步：批量拉取真实标签，先只看csv结果
    df = batch_fetch_real_labels(
        api_key=api_key,
        cif_dir="data/cif_files",
        dos_dir="data/dos_data",
        output_csv="data/real_labels.csv",
    )
    # 和graph_data.pt的合并（新增y_real/y_real_mask字段）先不做，
    # 等确认csv里的覆盖率和数值没问题，需要接入训练时再加回来。
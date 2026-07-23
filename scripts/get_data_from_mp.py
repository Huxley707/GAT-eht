from mp_api.client import MPRester
import os
output_dir = "data/cif_files"

with MPRester("J1uYn8CaW0o9oUo7S60H2mjJDVPDs0Qm") as mpr:

    ids=mpr.get_material_ids("VO2")
    string_ids = [str(mpid) for mpid in ids]


    # 批量搜索材料
    docs = mpr.materials.summary.search(
        material_ids=string_ids,
        fields=["structure", "material_id", "formula_pretty"]
    )

    print(f"找到 {len(docs)} 个材料")

    # 逐个保存为CIF文件
    success_count = 0
    for doc in docs:
        try:
            if doc.structure:
                # 使用material_id作为文件名
                filename = f"{doc.material_id}.cif"
                filepath = os.path.join(output_dir, filename)

                # 将结构保存为CIF格式
                doc.structure.to(filename=filepath)

                # 打印进度
                formula = getattr(doc, 'formula_pretty', '未知')
                print(f"已保存: {doc.material_id} ({formula}) -> {filepath}")
                success_count += 1
            else:
                print(f"警告: {doc.material_id} 没有结构数据")
        except Exception as e:
            print(f"保存 {doc.material_id} 时出错: {e}")

    print(f"\n下载完成! 成功保存 {success_count} 个CIF文件到 {output_dir} 文件夹")


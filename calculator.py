import json
import time
from geopy.geocoders import Nominatim


def get_position(INPUT_PATH, OUTPUT_PATH):
    # 输入输出文件名

    # 地理编码器
    geolocator = Nominatim(user_agent="sg_poi_latlng_batch", timeout=10)

    # 读取原文件
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i, item in enumerate(data):
        name = item["name"]

        # 若已有lat/lng则跳过
        if "lat" in item and "lng" in item:
            continue

        # 查询
        try:
            location = geolocator.geocode(f"{name}, Singapore")
            if location:
                item["lat"] = location.latitude
                item["lng"] = location.longitude
                print(f"[{i + 1}/{len(data)}] {name}: ({location.latitude}, {location.longitude})")
            else:
                item["lat"], item["lng"] = None, None
                print(f"[{i + 1}/{len(data)}] {name}: 未找到地理位置")
        except Exception as e:
            item["lat"], item["lng"] = None, None
            print(f"[{i + 1}/{len(data)}] {name}: 查询失败 - {e}")

        # Nominatim免费接口有请求间隔规定
        time.sleep(1.1)

    # 保存新文件
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n批量添加经纬度完成！已保存为 {OUTPUT_PATH}")

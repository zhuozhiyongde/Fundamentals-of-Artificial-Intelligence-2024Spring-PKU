#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author  :   Arthals
# @File    :   parallel.py
# @Time    :   2024/06/30 18:42:55
# @Contact :   zhuozhiyongde@126.com
# @Software:   Visual Studio Code

"""
parallel.py: 并行处理数据，将数据分成多个进程处理，加快处理速度。
"""

import json
import multiprocessing
import os


def find_match_lines(file_path):
    match_lines = []
    total_lines = -1
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if line.startswith("Match"):
                match_lines.append(i)
            total_lines = i
    total_lines += 1
    return match_lines, total_lines


def parallel_process(file_path, start_line, end_line, offset, cpu_id, output_dir):
    os.system(
        f"python preprocess.py {file_path} {start_line} {end_line} {offset} {cpu_id} {output_dir}"
    )


if __name__ == "__main__":
    file_name = "data"
    output_dir = "data-vec"

    os.makedirs(output_dir, exist_ok=True)
    file_path = f"data/{file_name}.txt"

    match_lines, total_lines = find_match_lines(file_path)
    total_matches = len(match_lines)
    print(len(match_lines), total_lines)
    # exit(0)
    num_cpus = multiprocessing.cpu_count()
    # num_cpus = 1
    match_per_process = 200
    chunk_size = num_cpus * match_per_process  # 每个进程处理200场比赛

    processes = []
    total_chunks = len(match_lines) // chunk_size + (
        1 if len(match_lines) % chunk_size != 0 else 0
    )

    for chunk_id in range(total_chunks):
        start_index = chunk_id * chunk_size

        for i in range(num_cpus):
            if start_index + i * match_per_process >= total_matches:
                break
            start_line = match_lines[start_index + i * match_per_process]
            end_line = (
                match_lines[start_index + (i + 1) * match_per_process] - 1
                if (start_index + (i + 1) * match_per_process) < total_matches
                else total_lines
            )

            offset = start_index + i * match_per_process

            p = multiprocessing.Process(
                target=parallel_process,
                args=(
                    file_path,
                    start_line,
                    end_line,
                    offset,
                    chunk_id * num_cpus + i,
                    output_dir,
                ),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
        processes.clear()  # 清除进程列表，准备下一轮

    # 合并所有进程生成的 JSON 文件
    total_out = []
    for chunk_id in range(total_chunks):
        for i in range(num_cpus):
            try:
                with open(
                    f"{output_dir}/count-{chunk_id * num_cpus + i}.json", "r"
                ) as f:
                    total_out += json.load(f)
                    # os.remove(f"{output_dir}/count-{chunk_id * num_cpus + i}.json")
            except FileNotFoundError:
                continue

    with open(f"data/{output_dir}-count.json", "w") as f:
        json.dump(total_out, f)

    # 校验 count.json 和 data-count-back.json 是否一致
    count = total_out

    with open("data/data-count-back.json", "r") as f:
        count_backup = json.load(f)

    print(len(count), len(count_backup))

    # 检查是否一致
    for i in range(len(count)):
        if count[i] != count_backup[i]:
            print(i)
            print(count[i], count_backup[i])
            break

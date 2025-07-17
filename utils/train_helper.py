# training time utils
# e.g., model save output_dir, etc

import os
import datetime
import pytz

def prepare_model_save_dir(args) -> str:
    """
    构造当前运行的 output_dir 并创建路径，返回该路径。

    使用 EST 时间作为时间戳，自动处理 note 和目录结构。
    model save 的dir 保存到arg 中, 是args.output_dir_this_run
    """
    # 使用美东时间
    tz = pytz.timezone("America/New_York")
    now = datetime.datetime.now(tz)
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    # 处理 note 后缀
    note_suffix = args.note.replace("/", "-") if args.note else "nonote"

    # 获取数据文件上层目录名，例如 personalized/data1/train.jsonl -> data1
    data_tag = os.path.basename(os.path.dirname(args.data_path))

    # 拼接 output 目录
    output_base_dir = os.path.join(
        args.output_dir,
        args.model_name_or_path.replace("/", "-"),
        data_tag,
        f"{timestamp}_{note_suffix}"
    )

    # 创建目录并赋值
    os.makedirs(output_base_dir, exist_ok=True)
    # args.output_dir_this_run = output_base_dir

    return output_base_dir

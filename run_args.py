import argparse
from util.consts import Tasks, Datasets, Models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("-batch", help="batch size", type=int, default=128)
    parser.add_argument("-epoch", help="train epoch", type=int, default=50)
    parser.add_argument("-slide_win", help="slide_win", type=int, default=5)
    parser.add_argument("-dim", help="Embeding dimension", type=int, default=64)
    parser.add_argument("-slide_stride", help="slide_stride", type=int, default=1)
    parser.add_argument(
        "-save_path_pattern", help="save path pattern", type=str, default=""
    )
    parser.add_argument(
        "-dataset", help="wadi / swat", type=Datasets, default=Datasets.swat
    )
    parser.add_argument("-device", help="cuda / cpu", type=str, default="cuda")
    parser.add_argument("-random_seed", help="random seed", type=int, default=0)
    parser.add_argument("-out_layer_num", help="outlayer num", type=int, default=1)
    parser.add_argument(
        "-out_layer_inter_dim", help="out_layer_inter_dim", type=int, default=64
    )
    parser.add_argument("-decay", help="decay", type=float, default=0)
    parser.add_argument("-val_ratio", help="val ratio", type=float, default=0.1)
    parser.add_argument("-topk", help="topk num", type=int, default=15)
    parser.add_argument("-report", help="best / val", type=str, default="best")
    parser.add_argument(
        "-load_model_path", help="trained model path", type=str, default=""
    )
    parser.add_argument("-model", help="trained model", type=Models, default=Models.gdn)
    parser.add_argument(
        "-task", help="training task", type=Tasks, default=Tasks.s_next_s
    )

    args = parser.parse_args()

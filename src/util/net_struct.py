import glob


def get_feature_map(dataset_name: str):
    """_summary_
    positions each column's name in returned list in order so given the index, name can be easily lookedup.
    list of columns is in list.txt file.
    Args:
        dataset_name (str): name of dataset.
    Returns:
        list: list of name of columns or sensor names.
    """
    list
    feature_file = open(f"./data/{dataset_name}/list.txt", "r")
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    return feature_list


def get_fully_connected_graph_struc(dataset_name: str) -> dict:
    """_summary_
    Creates a graph which is fully-connect. keies of return dict is columns' names.
    Args:
        dataset_name (str):

    Returns:
        dict:
    """
    feature_file = open(f"./data/{dataset_name}/list.txt", "r")

    struc_map = {}
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []

        for other_ft in feature_list:
            if other_ft is not ft:
                struc_map[ft].append(other_ft)

    return struc_map


def get_prior_graph_struc(dataset):
    feature_file = open(f"./data/{dataset}/features.txt", "r")

    struc_map = {}
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []
        for other_ft in feature_list:
            if dataset == "wadi" or dataset == "wadi2":
                # same group, 1_xxx, 2A_xxx, 2_xxx
                if other_ft is not ft and other_ft[0] == ft[0]:
                    struc_map[ft].append(other_ft)
            elif dataset == "swat":
                # FIT101, PV101
                if other_ft is not ft and other_ft[-3] == ft[-3]:
                    struc_map[ft].append(other_ft)

    return struc_map


if __name__ == "__main__":
    get_graph_struc()

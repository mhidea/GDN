# preprocess data
import numpy as np
import re
import pandas as pd


def get_most_common_features(target, all_features, max=3, min=3):
    res = []
    main_keys = target.split("_")

    for feature in all_features:
        if target == feature:
            continue

        f_keys = feature.split("_")
        common_key_num = len(list(set(f_keys) & set(main_keys)))

        if common_key_num >= min and common_key_num <= max:
            res.append(feature)

    return res


def build_net(target, all_features):
    # get edge_indexes, and index_feature_map
    main_keys = target.split("_")
    edge_indexes = [[], []]
    index_feature_map = [target]

    # find closest features(nodes):
    parent_list = [target]
    graph_map = {}
    depth = 2

    for i in range(depth):
        for feature in parent_list:
            children = get_most_common_features(feature, all_features)

            if feature not in graph_map:
                graph_map[feature] = []

            # exclude parent
            pure_children = []
            for child in children:
                if child not in graph_map:
                    pure_children.append(child)

            graph_map[feature] = pure_children

            if feature not in index_feature_map:
                index_feature_map.append(feature)
            p_index = index_feature_map.index(feature)
            for child in pure_children:
                if child not in index_feature_map:
                    index_feature_map.append(child)
                c_index = index_feature_map.index(child)

                edge_indexes[1].append(p_index)
                edge_indexes[0].append(c_index)

        parent_list = pure_children

    return edge_indexes, index_feature_map


def construct_data(data, feature_map, labels=0):
    res = []

    for feature in feature_map:
        if feature in data.columns:
            res.append(data.loc[:, feature].values.tolist())
        else:
            print(feature, "not exist in data")
    # append labels as last
    sample_n = len(res[0])

    if type(labels) == int:
        res.append([labels] * sample_n)
    elif len(labels) == sample_n:
        res.append(labels)

    return res


def build_loc_net(struc: dict, all_features: list, feature_map=[]) -> list:
    """Creates fully connected adjacency matrix.
    list of size (2 , Edges) list[0][i] is connected to list[1][i].

    Args:
        struc (dict): _description_
        all_features (list): _description_
        feature_map (list, optional): _description_. Defaults to [].

    Returns:
        list: list of size (2 , Edges).
    """
    index_feature_map = feature_map
    edge_indexes = [[], []]
    for node_name, node_list in struc.items():
        if node_name not in all_features:
            continue

        if node_name not in index_feature_map:
            index_feature_map.append(node_name)

        p_index = index_feature_map.index(node_name)
        for child in node_list:
            if child not in all_features:
                continue

            if child not in index_feature_map:
                print(f"error: {child} not in index_feature_map")
                # index_feature_map.append(child)

            c_index = index_feature_map.index(child)
            # edge_indexes[0].append(p_index)
            # edge_indexes[1].append(c_index)
            edge_indexes[0].append(c_index)
            edge_indexes[1].append(p_index)

    return edge_indexes


def findSensorActuator(dataFrame: pd.DataFrame, ignor_labels: list = None):
    actuators = []
    consts = {}
    sensors = []
    for col in dataFrame.columns:
        l = len(dataFrame[col].unique())
        if l == 2:
            if ignor_labels is None:
                actuators.append(col)
            elif l not in ignor_labels:
                actuators.append(col)
        elif l == 1:
            # print("const: ", col, " = ", dataFrame.iloc[0][col])
            consts[col] = dataFrame.iloc[0][col]
        else:
            sensors.append(col)
    print("#####################################")
    print("sensors count: ", len(sensors))
    print("actuators count: ", len(actuators))
    print("consts count: ", len(consts))
    print("consts: ", consts)
    print("#####################################")

    return sensors, actuators, consts

from collections import defaultdict
import json
import argparse
from tqdm import tqdm
import networkx as nx
from copy import deepcopy
from community import community_louvain


def input_data(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except:
        data = []
        for line in open(path, 'r'):
            data.append(json.loads(line))
    if type(data) == dict:
        data = [data]
    return data


def get_relation(text):
    if 'unrelated' in text:
        return False
    return True


def init_graph(g, data1, data2, data3, threshold):
    edge_list1 = []
    edge_list2 = []
    edge_list3 = []
    node_list = []
    for line in tqdm(data1):
        node_list.append(tuple(line['claim_id'][0]))
        node_list.append(tuple(line['claim_id'][1]))
    node_list = list(set(node_list))
    for node in node_list:
        g.add_node(node)
    tmp_map = {}
    tmp_num = 0
    for line in tqdm(data1):
        if ("corrcoef" in line.keys() and line['corrcoef'] > threshold) or get_relation(line['relation']):
            node_x = tuple(line['claim_id'][0])
            node_y = tuple(line['claim_id'][1])
            edge_list1.append(node_x[0])
            tmp_num += 1
            tmp_map[(node_x, node_y)] = 1
            tmp_map[(node_y, node_x)] = 1
    print("relation1 is True(A=1): ", tmp_num)
    tmp_num = 0
    tmp_num2 = 0
    for line in tqdm(data2):
        if ("corrcoef" in line.keys() and line['corrcoef'] > threshold) or get_relation(line['relation']):
            node_x = tuple(line['claim_id'][0])
            node_y = tuple(line['claim_id'][1])
            tmp_num += 1
            if (node_x, node_y) in tmp_map and tmp_map[(node_x, node_y)] == 1:
                tmp_num2 += 1
                edge_list2.append(node_x[0])
                tmp_map[(node_x, node_y)] = 2
                tmp_map[(node_y, node_x)] = 2
    print("relation2 is True(B=1): ", tmp_num)
    print("A=1 and B=1: ", tmp_num2)
    tmp_num = 0
    tmp_num2 = 0
    for line in tqdm(data3):
        # if (node_x, node_y) in tmp_map and tmp_map[(node_x, node_y)] == 2:
        #     if "corrcoef" in line.keys():
        #         if line['corrcoef'] < 0.0:
        #             import ipdb; ipdb.set_trace()
        if "corrcoef" in line.keys():
            if line['corrcoef'] > threshold:
                node_x = tuple(line['claim_id'][0])
                node_y = tuple(line['claim_id'][1])
                tmp_num += 1
                if (node_x, node_y) in tmp_map and tmp_map[(node_x, node_y)] == 2:
                    tmp_num2 += 1
                    edge_list3.append(node_x[0])
                    g.add_edge(node_x, node_y)
            continue
        if get_relation(line['relation']):
            node_x = tuple(line['claim_id'][0])
            node_y = tuple(line['claim_id'][1])
            tmp_num += 1
            if (node_x, node_y) in tmp_map and tmp_map[(node_x, node_y)] == 2:
                tmp_num2 += 1
                edge_list3.append(node_x[0])
                g.add_edge(node_x, node_y)
    print("relation3 is True(C=1): ", tmp_num)
    print("A=1 and B=1 and C=1: ", tmp_num2)
    return edge_list1, edge_list2, edge_list3


def remove_bridges(graph, bridges):
    components = list(nx.connected_components(graph))
    for bridge in bridges:
        delate_flag = False
        for component in components:
            if bridge[0] in component or bridge[1] in component:
                if len(component) > 3:
                    delate_flag = True
                break
        if delate_flag:
            graph.remove_edge(*bridge)
    return graph


def get_static_inf(edge_list1, edge_list2, edge_list3, components, st, ed):
    components_inf_dict = {}
    for component in components:
        new_component = list(component)
        if new_component[0][0] >= st and (ed < 0 or new_component[0][0] < ed):
            components_inf_dict[len(new_component)] = 0
    total_len = 0
    num = 0
    for component in components:
        new_component = list(component)
        if new_component[0][0] >= st and (ed < 0 or new_component[0][0] < ed):
            components_inf_dict[len(new_component)] += 1
            total_len += len(new_component)
            num += 1
    static_inf = []
    for i, j in components_inf_dict.items():
        static_inf.append((i, j))
    sorted(static_inf)
    print(f"response-st:{st} ed:{ed}")
    print("total claim class =", static_inf)
    print("total_num = ", num)
    print("avg_len = ", total_len / num)
    edge1_num = 0
    for item in edge_list1:
        if item >= st and (ed < 0 or item < ed):
            edge1_num += 1
    edge2_num = 0
    for item in edge_list2:
        if item >= st and (ed < 0 or item < ed):
            edge2_num += 1
    edge3_num = 0
    for item in edge_list3:
        if item >= st and (ed < 0 or item < ed):
            edge3_num += 1
    print(f"edge1:{edge1_num}   edge2:{edge2_num}   edge3:{edge3_num}")
    print("-------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path1', type=str)
    parser.add_argument('--data_path2', type=str, default='N')
    parser.add_argument('--data_path3', type=str, default='N')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--cluster_type', type=str)
    args = parser.parse_args()

    print(f"reading data1:{args.data_path1}...")
    data1 = input_data(args.data_path1)
    if len(args.data_path2) > 2:
        print(f"reading data2:{args.data_path2}...")
        data2 = input_data(args.data_path2)
    else:
        data2 = data1
    if len(args.data_path3) > 2:
        print(f"reading data3:{args.data_path3}...")
        data3 = input_data(args.data_path3)
    else:
        data3 = data1

    print("Building Edge-Biconnected Components Graph")
    g = nx.Graph()
    edge_list1, edge_list2, edge_list3 = init_graph(g, data1, data2, data3, args.threshold)
    assert args.cluster_type in ['tarjan', 'louvain']
    if args.cluster_type == 'tarjan':
        print("cluster_type: tarjan")
        bridge_list = list(nx.bridges(g))
        g = remove_bridges(g, bridge_list)
        components = list(nx.connected_components(g))
        components_list = []
        for component in components:
            components_list.append(list(component))
    else:
        print("cluster_type: louvain")
        partition = community_louvain.best_partition(graph=g, resolution=1.0)
        community = {}
        for node_id, community_id in partition.items():
            try:
                community[community_id].append(node_id)
            except:
                community[community_id] = []
                community[community_id].append(node_id)
        components_list = list(community.values())
        components = deepcopy(components_list)
    
    # import ipdb; ipdb.set_trace()
    print("--------------")
    print("graph info:")
    get_static_inf(edge_list1, edge_list2, edge_list3, components, 0, -1)
    max_response_id, min_response_id = 0, 100000000
    for component in components:
        new_component = list(component)
        min_response_id = min(min_response_id, new_component[0][0])
        max_response_id = max(max_response_id, new_component[0][0])
    range_step = (max_response_id // 10) - (min_response_id // 10) + 1
    print("[min_response_id, max_response_id] = ", min_response_id, max_response_id)
    for i in range(min_response_id, max_response_id+1, range_step):
        get_static_inf(edge_list1, edge_list2, edge_list3, components, i, i+range_step)
    print("--------------")

    with open(args.output_path, 'w') as f:
        json.dump(components_list, f, indent=4)
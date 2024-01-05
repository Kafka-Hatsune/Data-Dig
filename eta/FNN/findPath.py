import networkx as nx
import pandas as pd
from geopy.distance import geodesic

NODES_PATH = 'node_mid.csv'
EDGES_PATH = 'road.csv'
NODES_DF = pd.read_csv(NODES_PATH)
EDGES_DF = pd.read_csv(EDGES_PATH)

def create_graph():
    G = nx.Graph() # 新建一张图
    # 加点
    for index, row in NODES_DF.iterrows():
        G.add_node(row['id'])
    # 加边
    for index, row in EDGES_DF.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['cost'])
    return G

def find_nearest_node(coordinateX, coordinateY, FIELD_DELTA=0.03):
    """
    输入为待匹配点的坐标
    寻找最近匹配点,寻找一个距离给定点对最近的点,并返回编号
    """
    lb, rb = coordinateX - FIELD_DELTA, coordinateX + FIELD_DELTA
    db, ub = coordinateY - FIELD_DELTA, coordinateY + FIELD_DELTA
    condition = (NODES_DF['lon'] < rb) & (NODES_DF['lon'] > lb) & (NODES_DF['lat'] < ub) & (NODES_DF['lat'] > db)
    sub_nodes_df = NODES_DF[condition].copy()
    if sub_nodes_df.empty:
        print("empty!!!!!!!!!")
        return -1
    sub_nodes_df['dis'] = sub_nodes_df.apply(lambda row: geodesic((coordinateY, coordinateX), (row['lat'], row['lon'])).kilometers, axis=1)
    min_idx = sub_nodes_df['dis'].idxmin()
    return sub_nodes_df.loc[min_idx]['id']

def get_shortest_path(G, source, target):  
    return nx.dijkstra_path(G, source=source, target=target)

def get_coordinates(id):
    return NODES_DF.loc[id, 'lon'], NODES_DF.loc[id, 'lat']

def get_edge_info(node1, node2):
    """
    返回(id, cost)
    """
    result1 = EDGES_DF[(EDGES_DF['source']==node1) & (EDGES_DF['target']==node2)]
    result2 = EDGES_DF[(EDGES_DF['source']==node2) & (EDGES_DF['target']==node1)]
    if result2.empty:
        return result1.iloc[0, 0], result1.iloc[0, 5]
    else:
        return result2.iloc[0, 0], result2.iloc[0, 5]
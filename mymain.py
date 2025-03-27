from models.GDN import GDN
from util.consts import Tasks

m = GDN(edge_index_sets=None, embeding_dim=64, node_num=1, task=Tasks.next_label)

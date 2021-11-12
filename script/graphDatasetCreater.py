from graph_converter import graph_utils
#convertData2graph, visualize_graph
import csv

def csv2graphDataset(csv_files, include_names=False):
    """
    csv_files = {0:'1-1.csv', 1:'3-2.csv',
                     2:'5-3.csv', 3:'7-4.csv'}
    """
    obj_names_sets = []
    datasets = []
    for num in range(len(csv_files)):
        file_path = csv_files[num]
        with open(file_path) as f:
            csv_file = csv.reader(f)
            positions_data = [[float(v) for v in row] for row in csv_file]
        for row, position_data in enumerate(positions_data):
            graph, obj_names = convertData2graph(position_data, num, include_names)
            if graph is not None:
                datasets.append(graph)
                if len(obj_names)!=0:
                    obj_names_sets.append(obj_names)
    return datasets, obj_names_sets



if __name__ == '__main__':
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))+ "/experiment_data"
    csv_path_list = {0:base_dir+'/SI/work.csv'}
    datasets, obj_names_sets = csv2graphDataset(csv_path_list, include_names=True)

    for i, (graph, obj_names) in enumerate(zip(datasets, obj_names_sets)):
        visualize_graph(graph=graph, node_labels=obj_names, save_graph_name=None, show_graph=True)
        if i >2:
            break
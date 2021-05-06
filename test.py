import numpy as np
import torch
import sys
import copy
from config import *

device = torch.device("cuda")

def get_semantic_ndarray(data):
    tensor_device = torch.Tensor(data).to(device)
    return model.getSemantic(tensor_device).cpu().numpy()

def calculate_distance(x, transform_matrix):
    return np.sqrt(np.dot(np.dot(x, transform_matrix), x.transpose()))

def RESULT_LOGGER(result_list, message):
    result_list.append('{}\n'.format(message))
    print(message)

def gen_sematic_vec(train_map):
    semantic_center_map={}
    cov_inv_map={}
    cov_inv_diag_map={}
    sigma_identity_map={}
    distance_map={}

    for certain_class,train_data in train_map.items():
        raw_output=get_semantic_ndarray(train_data)
        semantic_center_map[certain_class] = np.mean(raw_output, 0)

        covariance_mat = np.cov(raw_output,rowvar=False,bias=True)
        cov_inv_map = np.linalg.pinv(covariance_mat)
        
        cov_inv_diag_mat=np.diagflat(1/(covariance_mat.diagonal()))
        cov_inv_diag_mat[cov_inv_diag_mat==np.inf]=0.0
        cov_inv_diag_map[certain_class]=cov_inv_diag_mat
        sigma = np.mean(np.diagflat(covariance_mat.diagonal()))
        sigma_identity_map[certain_class]=1/sigma*np.eye(covariance_mat.shape[0])

    distance_map['Maha'] = cov_inv_map
    distance_map['MahaDiag'] = cov_inv_diag_map
    distance_map['SigmaEye'] = sigma_identity_map

    return semantic_center_map, distance_map

def classify_evol(transform_map,semantic_center_map,semantic_vector,coef,coef_unknown):
    predicted_label=-1
    min_dist=float('inf')
    min_dist_recorded=float('inf')
    dists_known_I=[]
    if_known=False

    for certain_class in range(num_class):
        semantic_center=semantic_center_map[certain_class]
        dist = calculate_distance(semantic_vector-semantic_center, transform_map[certain_class])
        
        eyeMat=np.eye(semantic_center_map[certain_class].shape[0])
        dist_I = calculate_distance(semantic_vector-semantic_center, eyeMat)

        dists_known_I.append(dist_I)

        if dist < 3 * np.sqrt(semantic_vector.shape[0]) * coef:
            if_known=True

        if dist<min_dist:
            min_dist=dist
            predicted_label=certain_class

    mean_dist=np.mean(dists_known_I)
    min_dist=min(dists_known_I)

    if not if_known:
        # first unknown instance shows up
        if len(semantic_center_map.keys())==num_class:
            predicted_label = -1
        else:
            if_recorded = False
            recorded_unknowns = set(semantic_center_map.keys())-set(list(range(num_class)))
            for recorded_unknown_class in recorded_unknowns:
                semantic_center=semantic_center_map[recorded_unknown_class]
                dist = calculate_distance(semantic_vector-semantic_center, eyeMat)
                if dist <= coef_unknown * (min_dist+mean_dist)/2:
                    if_recorded=True
                    break

            if if_recorded:
                for recorded_unknown_class in recorded_unknowns:
                    semantic_center=semantic_center_map[recorded_unknown_class]
                    dist = calculate_distance(semantic_vector-semantic_center, eyeMat)
                    if dist < min_dist_recorded:
                        min_dist_recorded=dist
                        predicted_label=recorded_unknown_class
            else:
                predicted_label = -1

    return predicted_label

# include the one shot sample
def cal_acc_evol(train_map, test_map, unknown_test_map, distance = 'MahaDiag'):
    semantic_center_map_origin, distance_map = gen_sematic_vec(train_map)

    print('Using cluster plus mode')
    with open(zero_shot_path,'w') as f:
        tackled_test_data = np.concatenate((*(test_map.values()),), 0)
        tackled_label = np.concatenate((*map(lambda x: np.full([x[1].shape[0]], x[0], dtype = np.int64), test_map.items()),), 0)
        tackled_unknown_test_data = np.concatenate((*(unknown_test_map.values()),), 0)
        tackled_unknown_label = np.concatenate((*map(lambda x: np.full([x[1].shape[0]], x[0], dtype = np.int64), unknown_test_map.items()),), 0)
        test_samples = np.concatenate((tackled_test_data, tackled_unknown_test_data), 0)
        test_labels = np.concatenate((tackled_label, tackled_unknown_label), 0)
        predicted_semantics = get_semantic_ndarray(test_samples)

        coef_unknown=1
        coef = 0.15

        while coef <= 1.0:
            resultlines=[]
            RESULT_LOGGER(resultlines, 'Distance {} with coefficient {}'.format(distance,coef))

            # shuffle test samples
            indices = np.random.permutation(test_samples.shape[0])
            predicted_semantics = predicted_semantics[indices]
            test_labels = test_labels[indices]

            semanticMap = copy.deepcopy(semantic_center_map_origin)

            transform_map=distance_map[distance]

            num_known_total = tackled_test_data.shape[0]      # total number of known samples
            num_known_unknown = 0     # number of known samples discriminated as unknown

            conf = np.zeros([len(classes), len(classes)])

            num_unknown_unknown = 0   # number of unknown samples discriminated as unknown
            num_unknown_total = tackled_unknown_test_data.shape[0]    # total number of unknown samples
            new_class_instances_map = {}
            new_class_labels_count_map = {}

            new_class_index = num_class

            for certain_class, predicted_semantic in zip(test_labels, predicted_semantics): 
                predicted_label = classify_evol(transform_map,semanticMap,predicted_semantic,coef,coef_unknown)

                # known class
                if predicted_label in range(num_class):
                    if certain_class in range(num_class):
                        conf[certain_class, predicted_label] += 1
                # unknown class
                else:
                    if certain_class in range(num_class):
                        num_known_unknown += 1
                    else:
                        num_unknown_unknown += 1
                    
                    # new unknown class discriminated
                    if predicted_label == -1:
                        # initialize new semantic center
                        semanticMap[new_class_index] = predicted_semantic
                        new_class_instances_map[new_class_index] = [predicted_semantic]
                        new_class_labels_count_map[new_class_index] = { int(certain_class): 1 }
                        new_class_index += 1
                    # classified as newly recorded class
                    else:
                        # update semantic center
                        new_class_instances_map[predicted_label].append(predicted_semantic)
                        semanticMap[predicted_label]=np.mean(new_class_instances_map[predicted_label], axis=0)
                        new_class_labels_count_map[predicted_label][int(certain_class)] = new_class_labels_count_map[predicted_label].get(int(certain_class), 0) + 1

            for certain_class, test_data in test_map.items():
                RESULT_LOGGER(resultlines, "Accuracy(class:{}):{}".format(classes[certain_class],conf[certain_class,certain_class]/test_data.shape[0]))
            RESULT_LOGGER(resultlines, 'Seen Accuracy: {}'.format(np.trace(conf)/num_known_total))

            false_unknown = num_known_unknown/num_known_total
            RESULT_LOGGER(resultlines, 'False Unknown Rate: {}\n'.format(false_unknown))

            # FUR is not qualified
            if false_unknown > 0.2:
                coef+=0.02
                continue

            true_unknown = num_unknown_unknown/num_unknown_total
            RESULT_LOGGER(resultlines, 'True Unknown Rate: {}'.format(true_unknown))
            RESULT_LOGGER(resultlines, 'Total number of newly identified class: {}'.format(len(new_class_labels_count_map)))

            new_class_candidate_label_map = {}
            for new_class, labels_count_map in new_class_labels_count_map.items():
                class_labels_sorted = sorted(labels_count_map.items(), key=lambda x: x[1], reverse=True)
                new_class_candidate_label_map[new_class] = class_labels_sorted[0]
            
            for unknown_class, test_data in unknown_test_map.items():
                # element: (new_class_label, (candidate_label, count))
                isotopic_new_classes = list(filter(lambda x: x[1][0] == unknown_class, new_class_candidate_label_map.items()))
                if len(isotopic_new_classes) < 1:
                    RESULT_LOGGER(resultlines, 'Unknown class {} fails to identify'.format(unknown_class))
                    break

                dominant_new_classes = max(isotopic_new_classes, key= lambda x: x[1][1])
                RESULT_LOGGER(resultlines, 'Unknown Class {}'.format(unknown_class))
                RESULT_LOGGER(resultlines, '    Accuracy: {}'.format(dominant_new_classes[1][1]/test_data.shape[0]))
                RESULT_LOGGER(resultlines, '    Precision: {}'.format(
                    dominant_new_classes[1][1]/len(new_class_instances_map[dominant_new_classes[0]]))
                )

            RESULT_LOGGER(resultlines, '')

            # TUR is qualified, write results
            if true_unknown > 0.8:
                f.writelines(resultlines)
            else:
                break

            if coef<1.0:
                coef+=0.02

    print('{} cluster finished'.format('Evalution zero-shot'))

if __name__ == '__main__':
    dataset.split_unknown()
    train_map,test_map,unknown_test_map=dataset.get_train_test_maps()

    model_paths = []
    if len(sys.argv) > 1 :
        path = sys.argv[1]
        if os.path.isdir(path):
            model_paths = [os.path.join(path,x) for x in os.listdir(path)]
            version = list(filter(lambda x: not x == '', path.split('/')))[-1]
        elif os.path.isfile(path):
            model_paths.append(path)
    else:
        model_paths.append(model_path)

    if not os.path.isdir('results/'+version):
        os.mkdir('results/'+version)

    model.to(device)

    for model_path in model_paths:
        print('Loading model from {}'.format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        zero_shot_path = 'results/'+version+'/ZSL_'+os.path.split(model_path)[-1][:-4]+'.txt'
        print('With {} epochs training...'.format(checkpoint['epoch']))

        with torch.no_grad():
            model.eval()

            print('ZSL evaluation')
            cal_acc_evol(train_map,test_map,unknown_test_map)

    print('end')

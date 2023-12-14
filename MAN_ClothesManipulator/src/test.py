""" Testing for the Clothes Manipulator in Neural Turing Machines. """

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from MAN_ClothesManipulator.src.data_supplier import DataSupplier
from MAN_ClothesManipulator.src.model import MemoryBlock
from MAN_ClothesManipulator.src.optimized_data_supplier import OptimizedDataSupplier
from MAN_ClothesManipulator.src.task.clothesManipulatorTask import ClothesManipulatorModelTraining, \
    ClothesManipulatroTaskParams

TASKS = {
    'clothesManipulatorTask': (ClothesManipulatorModelTraining, ClothesManipulatroTaskParams),
}


def init_model(args):
    model_cls, params_cls = TASKS[args]
    params = params_cls()

    model = model_cls(params=params)  # Encapsuleted
    return model


def evaluate_models(model1, model2, data_supplier, distance):
    ten_correct_1 = 0
    twenty_correct_1 = 0
    thirty_correct_1 = 0
    fourty_correct_1 = 0
    fifty_correct_1 = 0

    ten_correct_2 = 0
    twenty_correct_2 = 0
    thirty_correct_2 = 0
    fourty_correct_2 = 0
    fifty_correct_2 = 0

    num_epochs = 10
    num_tests = 100

    with torch.no_grad():
        for k in range(num_epochs):
            print("***** Epoch: ", k + 1, "/", num_epochs, " *****")
            for j in range(num_tests):
                print("Test: ", j + 1, "/", num_tests)
                ok = False
                while not ok:
                    ok, q_id, t_id = data_supplier.get_next_pair_sample(
                        distance)  # data_supplier.get_next_pair_sample_exact(1)
                    if ok:
                        # MAN
                        q_dis_feat, t_dis_feat = data_supplier.get_disentangled_features(q_id, t_id)
                        control, manip_vectors = data_supplier.get_manipulation_vectors(q_id, t_id,
                                                                                        distance)  # input 151

                        target = torch.reshape(torch.from_numpy(t_dis_feat), (12, 1, 340))
                        query = torch.reshape(torch.from_numpy(q_dis_feat), (1, 12, 340))
                        net_inputs = torch.reshape(torch.from_numpy(manip_vectors),
                                                   (len(manip_vectors), 1, 151)).float()

                        inp_seq_len = net_inputs.size(0)
                        outp_seq_len, batch_size, _ = target.size()

                        # New sequence
                        model1.net.init_sequence_query(batch_size, query)

                        # Feed the sequence + delimiter
                        for i in range(inp_seq_len):
                            model1.net(net_inputs[i])

                        net_memory = model1.net.get_memory()
                        memory = net_memory.memory.numpy().reshape(4080)

                        t_one_hot = data_supplier.get_on_hot_label(t_id)
                        best_ids = data_supplier.find_x_ids_images_more_similiar(memory, 50)
                        for i in range(50):
                            curr_one_hot = data_supplier.get_on_hot_label(best_ids[i])
                            if (curr_one_hot == t_one_hot).all():
                                if i < 10:
                                    ten_correct_1 += 1
                                    twenty_correct_1 += 1
                                    thirty_correct_1 += 1
                                    fourty_correct_1 += 1
                                    fifty_correct_1 += 1
                                    print("distance: ", inp_seq_len)
                                    print("10: ", q_id, t_id)
                                elif i < 20:
                                    twenty_correct_1 += 1
                                    thirty_correct_1 += 1
                                    fourty_correct_1 += 1
                                    fifty_correct_1 += 1
                                elif i < 30:
                                    thirty_correct_1 += 1
                                    fourty_correct_1 += 1
                                    fifty_correct_1 += 1
                                elif i < 40:
                                    fourty_correct_1 += 1
                                    fifty_correct_1 += 1
                                else:
                                    fifty_correct_1 += 1
                                    print("distance: ", inp_seq_len)
                                    print("50: ", q_id, t_id)
                                break

                        # MODELLO AMAZON

                        q_dis_feat = torch.reshape(torch.from_numpy(q_dis_feat), (1, 4080)).cuda()
                        query_fused_feats = q_dis_feat
                        for i in range(inp_seq_len):
                            residual_feat = model2(net_inputs[i].cuda())
                            query_fused_feats = F.normalize(query_fused_feats + residual_feat)
                        query_fused_feats = query_fused_feats.cpu().numpy().reshape(4080)
                        best_ids = data_supplier.find_x_ids_images_more_similiar(query_fused_feats, 50)
                        for i in range(50):
                            curr_one_hot = data_supplier.get_on_hot_label(best_ids[i])
                            if (curr_one_hot == t_one_hot).all():
                                if i < 10:
                                    ten_correct_2 += 1
                                    twenty_correct_2 += 1
                                    thirty_correct_2 += 1
                                    fourty_correct_2 += 1
                                    fifty_correct_2 += 1
                                elif i < 20:
                                    twenty_correct_2 += 1
                                    thirty_correct_2 += 1
                                    fourty_correct_2 += 1
                                    fifty_correct_2 += 1
                                elif i < 30:
                                    thirty_correct_2 += 1
                                    fourty_correct_2 += 1
                                    fifty_correct_2 += 1
                                elif i < 40:
                                    fourty_correct_2 += 1
                                    fifty_correct_2 += 1
                                else:
                                    fifty_correct_2 += 1
                                break

    print("MAN: ")
    print("Top-10: ", (ten_correct_1 / (num_tests * num_epochs)) * 100, "%")
    print("Top-20: ", (twenty_correct_1 / (num_tests * num_epochs)) * 100, "%")
    print("Top-30: ", (thirty_correct_1 / (num_tests * num_epochs)) * 100, "%")
    print("Top-40: ", (fourty_correct_1 / (num_tests * num_epochs)) * 100, "%")
    print("Top-50: ", (fifty_correct_1 / (num_tests * num_epochs)) * 100, "%")

    print("AMAZON: ")
    print("Top-10: ", (ten_correct_2 / (num_tests * num_epochs)) * 100, "%")
    print("Top-20: ", (twenty_correct_2 / (num_tests * num_epochs)) * 100, "%")
    print("Top-30: ", (thirty_correct_2 / (num_tests * num_epochs)) * 100, "%")
    print("Top-40: ", (fourty_correct_2 / (num_tests * num_epochs)) * 100, "%")
    print("Top-50: ", (fifty_correct_2 / (num_tests * num_epochs)) * 100, "%")
    print("Done Evaluating!")


def visual_evaluate_models_step_by_step(model1, model2, data_supplier, distance, q_id, t_id):
    with torch.no_grad():
        # MAN
        q_dis_feat, t_dis_feat = data_supplier.get_disentangled_features(q_id, t_id)
        control, manip_vectors = data_supplier.get_manipulation_vectors(q_id, t_id, distance)  # input 151

        target = torch.reshape(torch.from_numpy(t_dis_feat), (12, 1, 340))
        query = torch.reshape(torch.from_numpy(q_dis_feat), (1, 12, 340))
        net_inputs = torch.reshape(torch.from_numpy(manip_vectors),
                                   (len(manip_vectors), 1, 151)).float()

        inp_seq_len = net_inputs.size(0)
        print("Distance: ", inp_seq_len)
        outp_seq_len, batch_size, _ = target.size()

        # New sequence
        model1.net.init_sequence_query(batch_size, query)

        # Feed the sequence + delimiter
        for i in range(inp_seq_len):
            model1.net(net_inputs[i])
            net_memory = model1.net.get_memory()
            memory = net_memory.memory.numpy().reshape(4080)
            best_ids = data_supplier.find_x_ids_images_more_similiar(memory, 10)

            fig = plt.figure()
            fig.suptitle('Visual Evaluation MAN - Manipulation: ' + str(i + 1), fontsize=16)
            img = data_supplier.get_image(q_id)[0]
            ax1 = fig.add_subplot(4, 5, 1)
            ax1.set_title('query image')
            ax1.imshow(normalizeImage(img.permute(1, 2, 0)))
            img = data_supplier.get_image(t_id)[0]
            ax2 = fig.add_subplot(4, 5, 6)
            ax2.set_title('target image')
            ax2.imshow(normalizeImage(img.permute(1, 2, 0)))

            img = data_supplier.get_image(best_ids[0])[0]
            ax3 = fig.add_subplot(4, 5, 11)
            ax3.set_title('top-10')
            ax3.imshow(normalizeImage(img.permute(1, 2, 0)))

            for i in range(1, 5):
                img = data_supplier.get_image(best_ids[i])[0]
                fig.add_subplot(4, 5, 11 + i).imshow(normalizeImage(img.permute(1, 2, 0)))
            for i in range(5):
                img = data_supplier.get_image(best_ids[i + 5])[0]
                fig.add_subplot(4, 5, 16 + i).imshow(normalizeImage(img.permute(1, 2, 0)))
            # plt.subplot_tool()
            plt.subplots_adjust(hspace=0.4)
            plt.show()

        # MODELLO AMAZON

        q_dis_feat = torch.reshape(torch.from_numpy(q_dis_feat), (1, 4080)).cuda()
        query_fused_feats = q_dis_feat
        for i in range(inp_seq_len):
            residual_feat = model2(net_inputs[i].cuda())
            query_fused_feats = F.normalize(query_fused_feats + residual_feat)
            current_result = query_fused_feats.cpu().numpy().reshape(4080)
            best_ids = data_supplier.find_x_ids_images_more_similiar(current_result, 10)

            fig = plt.figure()
            fig.suptitle('Visual Evaluation ADDE-M - Manipulation: ' + str(i + 1), fontsize=16)
            img = data_supplier.get_image(q_id)[0]
            ax1 = fig.add_subplot(4, 5, 1)
            ax1.set_title('query image')
            ax1.imshow(normalizeImage(img.permute(1, 2, 0)))
            img = data_supplier.get_image(t_id)[0]
            ax2 = fig.add_subplot(4, 5, 6)
            ax2.set_title('target image')
            ax2.imshow(normalizeImage(img.permute(1, 2, 0)))

            img = data_supplier.get_image(best_ids[0])[0]
            ax3 = fig.add_subplot(4, 5, 11)
            ax3.set_title('top-10')
            ax3.imshow(normalizeImage(img.permute(1, 2, 0)))

            for i in range(1, 5):
                img = data_supplier.get_image(best_ids[i])[0]
                fig.add_subplot(4, 5, 11 + i).imshow(normalizeImage(img.permute(1, 2, 0)))
            for i in range(5):
                img = data_supplier.get_image(best_ids[i + 5])[0]
                fig.add_subplot(4, 5, 16 + i).imshow(normalizeImage(img.permute(1, 2, 0)))
            # plt.subplot_tool()
            plt.subplots_adjust(hspace=0.4)
            plt.show()

    print("Done Visual Evaluating!")



def evaluate_model(model, data_supplier, distance):
    ten_correct = 0
    twenty_correct = 0
    thirty_correct = 0
    fourty_correct = 0
    fifty_correct = 0
    num_tests = 100

    with torch.no_grad():
        for j in range(num_tests):
            print(j, "/", num_tests)
            ok = False
            while not ok:
                ok, q_id, t_id = data_supplier.get_next_pair_sample(
                    distance)  # data_supplier.get_next_pair_sample_exact(1)
                if ok:
                    q_dis_feat, t_dis_feat = data_supplier.get_disentangled_features(q_id, t_id)
                    control, manip_vectors = data_supplier.get_manipulation_vectors(q_id, t_id, distance)  # input 151

                    target = torch.reshape(torch.from_numpy(t_dis_feat), (12, 1, 340))
                    query = torch.reshape(torch.from_numpy(q_dis_feat), (1, 12, 340))
                    net_inputs = torch.reshape(torch.from_numpy(manip_vectors), (len(manip_vectors), 1, 151)).float()

                    inp_seq_len = net_inputs.size(0)
                    outp_seq_len, batch_size, _ = target.size()

                    # New sequence
                    model.net.init_sequence_query(batch_size, query)

                    # Feed the sequence + delimiter
                    for i in range(inp_seq_len):
                        model.net(net_inputs[i])

                    net_memory = model.net.get_memory()
                    memory = net_memory.memory.numpy().reshape(4080)

                    t_one_hot = data_supplier.get_on_hot_label(t_id)
                    best_ids = data_supplier.find_x_ids_images_more_similiar(memory, 50)
                    for i in range(50):
                        curr_one_hot = data_supplier.get_on_hot_label(best_ids[i])
                        if (curr_one_hot == t_one_hot).all():
                            if i < 10:
                                ten_correct += 1
                                twenty_correct += 1
                                thirty_correct += 1
                                fourty_correct += 1
                                fifty_correct += 1
                            elif i < 20:
                                twenty_correct += 1
                                thirty_correct += 1
                                fourty_correct += 1
                                fifty_correct += 1
                            elif i < 30:
                                thirty_correct += 1
                                fourty_correct += 1
                                fifty_correct += 1
                            elif i < 40:
                                fourty_correct += 1
                                fifty_correct += 1
                            else:
                                fifty_correct += 1
                            break

    print("Top-10: ", (ten_correct / num_tests) * 100, "%")
    print("Top-20: ", (twenty_correct / num_tests) * 100, "%")
    print("Top-30: ", (thirty_correct / num_tests) * 100, "%")
    print("Top-40: ", (fourty_correct / num_tests) * 100, "%")
    print("Top-50: ", (fifty_correct / num_tests) * 100, "%")
    print("Done Evaluating!")


def normalizeImage(img):
    numpy_img = img.numpy()
    max = numpy_img.max()
    min = numpy_img.min()
    return (numpy_img - min) / (max - min)


def visual_evaluate_model(model, data_supplier, distance):
    num_tests = 1

    with torch.no_grad():
        for j in range(num_tests):
            print(j, "/", num_tests)
            ok = False
            while not ok:
                # ok, q_id, t_id = data_supplier.get_next_pair_sample(distance)  # data_supplier.get_next_pair_sample_exact(1)
                ok, q_id, t_id = True, 4475, 12580
                if ok:
                    q_dis_feat, t_dis_feat = data_supplier.get_disentangled_features(q_id, t_id)
                    control, manip_vectors = data_supplier.get_manipulation_vectors(q_id, t_id, distance)  # input 151

                    target = torch.reshape(torch.from_numpy(t_dis_feat), (12, 1, 340))
                    query = torch.reshape(torch.from_numpy(q_dis_feat), (1, 12, 340))
                    net_inputs = torch.reshape(torch.from_numpy(manip_vectors), (len(manip_vectors), 1, 151)).float()

                    inp_seq_len = net_inputs.size(0)
                    print("Distance: ", inp_seq_len)
                    outp_seq_len, batch_size, _ = target.size()

                    # New sequence
                    model.net.init_sequence_query(batch_size, query)

                    # Feed the sequence + delimiter
                    for i in range(inp_seq_len):
                        model.net(net_inputs[i])

                    net_memory = model.net.get_memory()
                    memory = net_memory.memory.numpy().reshape(4080)

                    best_ids = data_supplier.find_x_ids_images_more_similiar(memory, 10)

                    fig = plt.figure()
                    fig.suptitle('Visual Evaluation MAN', fontsize=16)
                    img = data_supplier.get_image(q_id)[0]
                    ax1 = fig.add_subplot(4, 5, 1)
                    ax1.set_title('query image')
                    ax1.imshow(normalizeImage(img.permute(1, 2, 0)))
                    img = data_supplier.get_image(t_id)[0]
                    ax2 = fig.add_subplot(4, 5, 6)
                    ax2.set_title('target image')
                    ax2.imshow(normalizeImage(img.permute(1, 2, 0)))

                    img = data_supplier.get_image(best_ids[0])[0]
                    ax3 = fig.add_subplot(4, 5, 11)
                    ax3.set_title('top-10')
                    ax3.imshow(normalizeImage(img.permute(1, 2, 0)))

                    for i in range(1, 5):
                        img = data_supplier.get_image(best_ids[i])[0]
                        fig.add_subplot(4, 5, 11 + i).imshow(normalizeImage(img.permute(1, 2, 0)))
                    for i in range(5):
                        img = data_supplier.get_image(best_ids[i + 5])[0]
                        fig.add_subplot(4, 5, 16 + i).imshow(normalizeImage(img.permute(1, 2, 0)))
                    # plt.subplot_tool()
                    plt.subplots_adjust(hspace=0.4)
                    plt.show()

    print("Done Visual Evaluating!")


def main():
    # Initialize arguments
    file_root = '../splits/Shopping100k'
    img_root_path = '../Shopping100k/Images'
    dis_feat_root = '../disentangledFeaturesExtractor'
    mode = 'test'
    task = "clothesManipulatorTask"

    # Initialize the model
    model1 = init_model(task)  # MODEL
    data_supplier = DataSupplier(file_root, img_root_path, dis_feat_root, mode, False)  # <= N (N grande)
    # data_supplier = OptimizedDataSupplier(file_root, img_root_path, dis_feat_root, mode, False) # == N (N piccolo)

    path = '../nets/700_200.pth'
    model1.net.load_state_dict(torch.load(path))
    # evaluate_model(model1, data_supplier, 1)
    #visual_evaluate_model(model1, data_supplier, 8)

    attr_num = np.loadtxt("../splits/Shopping100k/attr_num.txt", dtype=int)
    # np.array([16, 17, 19, 14, 10, 15, 2, 11, 16, 7, 9, 15])
    model2 = MemoryBlock(attr_num)
    path_pretrained_memory = '../models/Shopping100k/memory_best.pkl'
    model2.load_state_dict(torch.load(path_pretrained_memory))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model2.cuda()
    model2.eval()
    # evaluate_models(model1, model2, data_supplier, 8)
    # 12956, 17646 8
    # 2345, 4667 5
    # 5793, 12682 6
    # 4475, 12580 8
    visual_evaluate_models_step_by_step(model1, model2, data_supplier, 8, 2345, 4667)


if __name__ == '__main__':
    main()

"""

NET 300_200

Top-10:  30.0%
Top-20:  42.0%
Top-30:  47.0%
Top-40:  51.0%
Top-50:  52.0%

NET 500_200

Top-10:  38.0%
Top-20:  50.0%
Top-30:  54.0%
Top-40:  60.0%
Top-50:  64.0%


NET 500_200 on only 1 distance test images

Top-10:  43.0 %
Top-20:  51.0 %
Top-30:  59.0 %
Top-40:  64.0 %
Top-50:  70.0 %

"""

"""
Test:  100 / 100
MAN: 
Top-10:  40.0 %
Top-20:  50.0 %
Top-30:  55.00000000000001 %
Top-40:  56.99999999999999 %
Top-50:  60.0 %

AMAZON: 
Top-10:  36.0 %
Top-20:  46.0 %
Top-30:  55.00000000000001 %
Top-40:  57.99999999999999 %
Top-50:  63.0 %

MAN: 
Top-10:  40.0 %
Top-20:  50.0 %
Top-30:  59.0 %
Top-40:  61.0 %
Top-50:  63.0 %

AMAZON: 
Top-10:  46.0 %
Top-20:  55.00000000000001 %
Top-30:  57.99999999999999 %
Top-40:  62.0 %
Top-50:  67.0 %

MAN: 
Top-10:  28.000000000000004 %
Top-20:  43.0 %
Top-30:  55.00000000000001 %
Top-40:  60.0 %
Top-50:  62.0 %

AMAZON: 
Top-10:  36.0 %
Top-20:  51.0 %
Top-30:  60.0 %
Top-40:  65.0 %
Top-50:  66.0 %

"""

"""
100 test samples with distance <= 8

MAN: 
Top-10:  33.0 %
Top-20:  42.0 %
Top-30:  49.0 %
Top-40:  54.0 %
Top-50:  55.00000000000001 %

AMAZON: 
Top-10:  2.0 %
Top-20:  2.0 %
Top-30:  3.0 %
Top-40:  3.0 %
Top-50:  3.0 %

900 test samples with distance <= 8

MAN: 
Top-10:  33.0 %
Top-20:  44.77777777777778 %
Top-30:  52.44444444444445 %
Top-40:  56.44444444444444 %
Top-50:  60.77777777777777 %

AMAZON: 
Top-10:  2.3333333333333335 %
Top-20:  4.111111111111112 %
Top-30:  5.333333333333334 %
Top-40:  6.222222222222222 %
Top-50:  7.222222222222221 %

***** 1000 test samples with distance <= 8 *****

MAN:
Top-10:  33.0 %
Top-20:  45.0 %
Top-30:  52.0 %
Top-40:  56.0 %
Top-50:  60.0 %

AMAZON: 
Top-10:  2.0 %
Top-20:  4.0 %
Top-30:  5.0 %
Top-40:  6.0 %
Top-50:  7.0 %

"""

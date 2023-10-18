""" Testing for the Clothes Manipulator in Neural Turing Machines. """

import torch
import matplotlib.pyplot as plt

from MAN_ClothesManipulator.src.data_supplier import DataSupplier
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
                ok, q_id, t_id = data_supplier.get_next_pair_sample(distance)  # data_supplier.get_next_pair_sample_exact(1)
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
                ok, q_id, t_id = data_supplier.get_next_pair_sample(distance)  # data_supplier.get_next_pair_sample_exact(1)
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
    model = init_model(task)  # MODEL
    # data_supplier = DataSupplier(file_root, img_root_path, dis_feat_root, mode, False)
    data_supplier = OptimizedDataSupplier(file_root, img_root_path, dis_feat_root, mode, False)

    path = '../nets/500_200_mix_1_3.pth'
    model.net.load_state_dict(torch.load(path))
    # evaluate_model(model, data_supplier, 1)
    visual_evaluate_model(model, data_supplier, 1)


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

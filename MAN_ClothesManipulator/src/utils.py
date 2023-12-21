import pickle
import numpy as np

with open('../multi_manip/split_index.obj', 'rb') as fp:
    split_index = pickle.load(fp)
    fp.close()
with open('../multi_manip/cut_index.obj', 'rb') as fp:
    cut_index = pickle.load(fp)
    fp.close()


def get_idx_label(label_vector, attr_num):
    """
    convert the one-hot attribute labels to the label that indicate GT's position for each attribute
    Args:
        label_vector: one-hot labels
        attr_num: 1-D list of numbers of attribute values for each attribute
    """
    start_idx = 0
    labels = []
    for i in attr_num:
        sub_labels = label_vector[start_idx:start_idx + i]
        if sum(sub_labels) > 0:
            assert sum(sub_labels) == 1
            labels.append(np.argmax(sub_labels))
        else:
            labels.append(-1)  # missing label
        start_idx += i
    return np.array(labels)


def listify_manip(multi_manip):
    Nsplit = np.split(multi_manip, split_index[:-1])
    manip_array = np.zeros((12, 151), dtype=int)

    for j in range(manip_array.shape[0]):
        start, end = cut_index[j][0], cut_index[j][1]
        manip_array[j][start:end] = Nsplit[j]

    manip_list = manip_array[~np.all(manip_array == 0, axis=1)]

    manip_list = manip_list[np.random.permutation(len(manip_list))]

    return manip_list


def split_labels(label, attr_num):
    """
    split the whole one-hot label into separate one-hot labels w.r.t attribute
    """
    labels = []
    start_idx = 0
    for i in attr_num:
        sub_labels = label[start_idx:start_idx + i]
        labels.append(sub_labels)
        start_idx += i
    return labels


def get_target_attr(indicator, attr_num):
    """
    given the indicator vector, return the target attribute that need to be changed
    """
    assert 1 in indicator  # sanity check, ensure target attribute exists
    start_idx = 0
    for i, val_num in enumerate(attr_num):
        sub_label = indicator[start_idx:start_idx + val_num]
        if 1 in sub_label:
            return i
        else:
            start_idx += val_num
    return -1


def compute_DCG(scores):
    """
    compute the Discounteed Cumulative Gain. Check the equation in the paper
    """
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)


def compute_NDCG(rank_scores):
    """
    compute the Normalized Discounteed Cumulative Gain. Check the equation in the paper
    """
    dcg = compute_DCG(rank_scores)
    idcg = compute_DCG(np.ones_like(rank_scores))
    ndcg = dcg / idcg
    return ndcg


ATTRIBUTES = {
    0: "Category",
    1: "Collar",
    2: "Color",
    3: "Fabric",
    4: "Fastening",
    5: "Fit",
    6: "Gender",
    7: "Neckline",
    8: "Pattern",
    9: "Pocket",
    10: "Sleeve",
    11: "Sport",
}

ATTRIBUTES_VALUES = {
    0: ["Coat",
        "Jacket",
        "Suit",
        "Shirt",
        "T-shirt",
        "Jumper",
        "Shorts",
        "Trouser",
        "Jean",
        "Swimming",
        "Jumpsuit",
        "Pyjamas",
        "Tracksuit Bottoms",
        "Tracksuit",
        "Skirt",
        "Dress"],
    1: ["Buttondown",
        "Cutaway",
        "High",
        "Hood",
        "Kent",
        "Lapel",
        "Lined",
        "Mandarin",
        "Polo",
        "Round",
        "Shawl",
        "Turndown",
        "V-neck",
        "Peter Pan",
        "Volant",
        "Shirt",
        "Chin"],
    2: ["Maroon",
        "Black",
        "White",
        "Charcoal",
        "Indigo",
        "Navy",
        "Pink",
        "Light blue",
        "Blue",
        "Beige",
        "Orange",
        "Yellow",
        "Red",
        "Purple",
        "Brown",
        "Olive",
        "Green",
        "Gray",
        "Multi"],
    3: ["Canvas",
        "Crocheted",
        "Denim",
        "Fleece",
        "Hardshell",
        "Jersey",
        "Jersey Lace",
        "Lace",
        "Mesh",
        "Mesh Jersey",
        "Rib",
        "Softshell",
        "Sweat",
        "Leather"],
    4: ["Zip",
        "Back closure",
        "Belt",
        "Button",
        "Covered",
        "Hook",
        "Laces",
        "No closure",
        "Front",
        "Others"],
    5: ["Bootcut",
        "Flared",
        "High waist",
        "Jeggings",
        "Large",
        "Loose",
        "Low",
        "Oversize",
        "Regular",
        "Skinny",
        "Slim",
        "Small",
        "Straight",
        "Tailered",
        "Tapered"
        ],
    6: ["Male",
        "Female"],
    7: ["Boat",
        "Backless",
        "Cache-coeur",
        "Henley",
        "Low v-neck",
        "Low round",
        "Off-the-shoulder",
        "Round",
        "Square",
        "V-neck",
        "Envelope"],
    8: ["Animal",
        "Burnout",
        "Camouflage",
        "Checked",
        "Colour gradient",
        "Colourful",
        "Floral",
        "Herringbone",
        "Marl",
        "Paisley",
        "Photo",
        "Pinstriped",
        "Plain",
        "Polka dot",
        "Print",
        "Striped"],
    9: ["Side",
        "Cargo",
        "Flap",
        "Napoleon",
        "Belt",
        "Sleeve",
        "Zip"],
    10: ["3/4",
         "Spaghetti",
         "Sleeveless",
         "Elbow",
         "Extra Long",
         "Extra Short",
         "Long",
         "Short",
         "Strapless"],
    11: ["Skii",
         "Basketball",
         "Football",
         "Golf",
         "Handball",
         "Hiking",
         "Climbing",
         "Running",
         "Snowboard",
         "Swim",
         "Tennis",
         "Training",
         "Yoga",
         "Urban",
         "Dance"]
}

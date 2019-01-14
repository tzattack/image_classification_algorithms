import os, time, json
# Set log level
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
import numpy as np

batch_size = 32


def get_label():
    '''
        read label file
    '''
    label_map_path = "./data_train/labels.txt"
    label_map_file = open(label_map_path)
    label_map = {}
    for line_number, label in enumerate(label_map_file.readlines()):
        label_map[line_number] = label[:-1]
        line_number += 1
    label_map_file.close()

    return label_map

def get_dataset(filename):
    '''
        read tfrecord files
    '''
    def parse_exmp(serial_exmp):
        feats = tf.parse_single_example(serial_exmp, features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.string),
            'image/filepath': tf.FixedLenFeature([], tf.string),
            })
        image_data = tf.cast(feats['image/encoded'], tf.string)
        image_data = tf.image.decode_image(image_data)
        label = tf.cast(feats['image/class/label'], tf.string)
        filepath = tf.cast(feats['image/filepath'], tf.string)
        return image_data, label, filepath

    dataset = tf.data.TFRecordDataset(filename)

    return dataset.map(parse_exmp)

def predict(filename):
    # get label
    label_map = get_label()
    # read tfrecord file by tf.data
    dataset = get_dataset(filename)
    # dataset = dataset.batch(1).repeat(1)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    # image_batch, file_batch = tf.train.batch([image_data, filepath],batch_size=8)

    # session
    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        t1 = time.time()
        # image_data_list, filepath_list, res = [], [], []
        res = []
        i = 0
        try:
            while True:
                # 通过session每次从数据集中取值
                [_image, _label, _filepath] = sess.run(fetches=next_element)
                i += 1
                _label = str(_label, encoding='utf-8')
                _filepath = str(_filepath, encoding='utf-8')
                _image = tf.divide(_image, 255)
                _image = tf.image.resize_images(_image, [331, 331])
                _image = sess.run(_image)
                _image = np.asarray([_image])
                _image = _image.reshape(-1, 331, 331, 3)
                # print(_image)

                with tf.device('/gpu:7'):
                    predictions = inference_session.run(output_layer, feed_dict={input_layer: _image})
                    # print(predictions)
                predictions = np.squeeze(predictions)

                overall_result = sess.run(tf.argmax(predictions))
                predict_result = label_map[overall_result].split(":")[-1]

                if predict_result == 'unknown': continue

                content = {}
                content['prob'] = str(np.max(predictions))
                content['label'] = predict_result
                content['filepath'] = _filepath
                res.append(content)

        except tf.errors.OutOfRangeError:
            t2 = time.time()
            print("average speed: {}s/image".format((t2 - t1) / i))
    return res


if __name__ == '__main__':

    # read model and load model graph
    model_dir = "./model"
    model = "nasnet_large_v1.pb"
    model_path = os.path.join(model_dir, model)
    model_graph = tf.Graph()
    with model_graph.as_default():
        with tf.gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
            input_layer = model_graph.get_tensor_by_name("input:0")
            output_layer = model_graph.get_tensor_by_name('final_layer/predictions:0')

    # Session Config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    inference_session = tf.Session(graph = model_graph, config=config)

    # Initialize session
    initializer = np.zeros([1, 331, 331, 3])
    inference_session.run(output_layer, feed_dict={input_layer: initializer})

    # Prediction
    file_list = []
    processed_files = []

    for path, dir, files in os.walk('./model_output/processed_files'):
        for file in files:
            processed_files.append(file.split('_')[0]+'.tfrecord')
    while True:
        for path, dir, files in os.walk("./input_data"):
            for file in files:
                if file == '.DS_Store': continue
                if file in processed_files: continue
                print("Reading file {}".format(file))
                file_path = os.path.join('./input_data', file)
                file_list.append(file_path)
                res = predict(file_path)
                processed_files.append(file)

                with open('./model_output/processed_files/{}_{}_processed_files.json'.format(file.split('.')[0], model.split('.')[0]), 'w') as f:
                    f.write(json.dumps(processed_files))

                with open('./model_output/classify_result/{}_{}_classify_result.json'.format(file.split('.')[0], model.split('.')[0]), 'w') as f:
                    f.write(json.dumps(res, indent=4, separators=(',',':')))

        time.sleep(1)

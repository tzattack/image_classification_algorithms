import os, time, json, multiprocessing, sys, random
# Set log level
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

batch_size = 16

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_label():
    '''read label file'''
    label_map_path = "./data_train/labels.txt"
    label_map_file = open(label_map_path)
    label_map = {}
    for line_number, label in enumerate(label_map_file.readlines()):
        label_map[line_number] = label[:-1]
        line_number += 1
    label_map_file.close()

    return label_map

def get_dataset(filename):
    import tensorflow as tf
    import numpy as np

    def parse_exmp(serial_exmp):
        feats = tf.parse_single_example(serial_exmp, features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.string),
            'image/filepath': tf.FixedLenFeature([], tf.string),
            })
        image_data = tf.cast(feats['image/encoded'], tf.string)
        image_data = tf.image.decode_jpeg(image_data)
        image_data = tf.image.resize_images(image_data, [331, 331])
        image_data = tf.divide(image_data, 255)
        label = tf.cast(feats['image/class/label'], tf.string)
        filepath = tf.cast(feats['image/filepath'], tf.string)
        return image_data, label, filepath

    # extract data
    dataset = tf.data.TFRecordDataset(filename, num_parallel_reads=8)
    dataset = dataset.map(parse_exmp)
    dataset = dataset.batch(batch_size)
    return dataset

def predict(process_id, filename, inference_sess, input_layer, output_layer):
    import tensorflow as tf
    import numpy as np
    # get label
    label_map = get_label()
    # read tfrecord file by tf.data
    dataset = get_dataset(filename)
    # dataset.apply(tf.contrib.data.prefetch_to_device("/gpu:0"))
    # load data
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()

    result = []
    content = {}
    count = 0
    # session
    with tf.Session() as sess:
        tf.global_variables_initializer()
        t1 = time.time()
        try:
            while True:
                [_image, _label, _filepath] = sess.run(fetches=features)
                _image = np.asarray([_image])
                _image = _image.reshape(-1, 331, 331, 3)

                predictions = inference_sess.run(output_layer, feed_dict={input_layer: _image})
                predictions = np.squeeze(predictions)

                # res = []
                # start = time.time()
                for i, pred in enumerate(predictions):
                    count += 1
                    overall_result = np.argmax(pred)
                    predict_result = label_map[overall_result].split(":")[-1]

                    if predict_result == 'unknown': continue

                    content['prob'] = str(np.max(pred))
                    content['label'] = predict_result
                    content['filepath'] = str(_filepath[i], encoding='utf-8')
                    result.append(content)
                # end = time.time()
                # print("process time: {}s".format(end-start))
        except tf.errors.OutOfRangeError:
            t2 = time.time()
            print("GPU {}\t{}\t{} images\tspeed: {}s".format(process_id, filename, count, (t2-t1)/count))
    return result

def write_process_file(file, model, processed_files):
    with open('./model_output/processed_files/{}_{}_processed_files.json'.format(file.split('.')[0], model.split('.')[0]), 'w') as f:
        f.write(json.dumps(processed_files))

def check_processed_files():
    import tensorflow as tf

    processed_files = []
    if not tf.gfile.Exists("./model_output/processed_files"):
        tf.gfile.MakeDirs("./model_output/processed_files")
    for path, dir, files in os.walk('./model_output/processed_files'):
        for file in files:
            processed_files.append(file.split('_')[0]+'.tfrecord')
    return processed_files

def write_classify_result(file, model, result):
    with open('./model_output/classify_result/{}_{}_classify_result.json'.format(file.split('.')[0], model.split('.')[0]), 'w') as f:
        f.write(json.dumps(result, indent=4, separators=(',',':')))

def set_locker():
    while True:
        with open('state.txt', 'r') as f:
            state = f.read()
        if state == 'reading':
            time.sleep(1)
        else:
            break

    with open('state.txt', 'w+') as f:
        f.write('reading')

def set_unlocker():
    with open("./state.txt", 'w+') as f:
        f.write('done')

def main(gpu_num):
    import tensorflow as tf
    import numpy as np

    # read model and load model graph
    model_dir = "./model"
    model = "nasnet_large_v1.pb"
    model_path = os.path.join(model_dir, model)
    model_graph = tf.Graph()
    with model_graph.as_default():
        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
            input_layer = model_graph.get_tensor_by_name("input:0")
            output_layer = model_graph.get_tensor_by_name('final_layer/predictions:0')

    config1 = tf.ConfigProto(
        # device_count={"GPU": 1},
        inter_op_parallelism_threads = 1,
        intra_op_parallelism_threads = 1,
        # log_device_placement=True,
    )
    config2 = tf.ConfigProto()
    config = config1
    config.gpu_options.allow_growth = False
    inference_session = tf.Session(graph=model_graph, config=config)

    # Initialize session
    initializer = np.zeros([1, 331, 331, 3])
    inference_session.run(output_layer, feed_dict={input_layer: initializer})
    print("GPU {} is ready.".format(gpu_num))

    time.sleep(int(gpu_num))

    file_list = []

    while True:
        processed_files = check_processed_files()
        # start = time.time()
        # set_locker()

        for path, dir, files in os.walk("./data_input"):
            files = files
            # file = files[0]

        # print("GPU {}".format(gpu_num))
        if '.DS_Store' in files: files.remove('.DS_Store')
        files = list(set(files).difference(set(processed_files)))
        files.sort()
        # print("GPU{}: ".format(gpu_num), files)
        if files == []:
            time.sleep(1)
            print("GPU {} waiting for new file...".format(gpu_num))
            # set_unlocker()
            continue
        else:
            file = files[0]
        # print("Reading file {}".format(file))
        processed_files.append(file)
        write_process_file(
            file=file,
            model=model,
            processed_files=processed_files
        )
        # print("GPU{} pf:".format(gpu_num), len(processed_files))

        # set_unlocker()

        file_path = os.path.join('./data_input', file)
        file_list.append(file_path)
        # stop = time.time()
        # print("process time:", (stop-start))
        # time.sleep(5)

        result = predict(
            process_id=gpu_num,
            filename=file_path,
            inference_sess=inference_session,
            input_layer=input_layer,
            output_layer=output_layer
        )

        write_classify_result(
            file=file,
            model=model,
            result=result
        )

if __name__ == '__main__':
    gpu_num = sys.argv[1]
    # set_unlocker()
    print("using gpu {}".format(gpu_num))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    main(gpu_num)

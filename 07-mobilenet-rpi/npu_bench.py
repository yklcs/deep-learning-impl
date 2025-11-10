from hailo_platform import __version__
from multiprocessing import Process, Queue
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, ConfigureParams,
 InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)
import time
import numpy as np 

    
def send(configured_network, images_list, num_images):
    vstreams_params = InputVStreamParams.make_from_network_group(configured_network, quantized=False, format_type=FormatType.FLOAT32)
    configured_network.wait_for_activation(100)
    print('Performing inference on input images...\n')
    with InputVStreams(configured_network, vstreams_params) as vstreams:
        for i in range(num_images):
            for j, vstream in enumerate(vstreams):
                data = np.expand_dims(images_list[i % 100], axis=0)
                vstream.send(data)                
               

def recv(configured_network, write_q, num_images):
    vstreams_params = OutputVStreamParams.make_from_network_group(configured_network, quantized=False, format_type=FormatType.FLOAT32)
    configured_network.wait_for_activation(100)

    with OutputVStreams(configured_network, vstreams_params) as vstreams:
        for _ in range(num_images):
            values = []
            for vstream in vstreams:
                data = vstream.recv()
                values.append(data)            
            write_q.put(values)


# Loading compiled HEFs to device:
hef = HEF("mobilenet_v2_1.0.hef")
height, width, channels = hef.get_input_vstream_infos()[0].shape
devices = Device.scan()

imgs = [np.zeros((height, width, channels), dtype=np.float32) for _ in range(300)]
num_images = 300

inputs = hef.get_input_vstream_infos()
outputs = hef.get_output_vstream_infos()

with VDevice(device_ids=devices) as target:
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()
    queue = Queue()  # outputs are stored in this queue 
    
    for layer_info in inputs:
        print('Input  layer: {} {}'.format(layer_info.name, layer_info.shape)) 

    for layer_info in outputs:
        print('Output layer: {} {}'.format(layer_info.name, layer_info.shape))


    send_process = Process(target=send, args=(network_group, imgs, num_images))
    recv_process = Process(target=recv, args=(network_group, queue, num_images))

    start_time = time.time()
    recv_process.start()
    send_process.start()

    with network_group.activate(network_group_params):
        num_out = 0
        while num_out < num_images:
            if not queue.empty():
                output = queue.get(0)
                num_out = num_out + 1
        
        recv_process.join()
        send_process.join()


    end_time = time.time()
    print('Inference was successful!\n')
    print('-------------------------------------')
    print(' Infer Time:      {:.3f} sec'.format(end_time - start_time))
    print(' Average FPS:     {:.3f}'.format(num_images/(end_time - start_time)))
    print('-------------------------------------')
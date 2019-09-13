import numpy as np
import tensorflow as tf
import cv2 
from firebase import firebase
from multiprocessing import Process, Value

# Read the graph.
with tf.gfile.FastGFile('inference_graph/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

base = "https://******.firebaseio.com/"
global firebases
a,b,c,d = 10,10,10,10;
def dothis():
    global firebase_god,firebase_bourn,firebase_nice,firebase_gat,goodday,bournvita,gatorade,nicetime
    cap = cv2.VideoCapture(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1240)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)
    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        try:
            while True:
                count = 0
                bournvita = 0;
                goodday = 0;
                gatorade = 0;
                nicetime = 0;
                global x,y,right,bottom
            # Read and preprocess an image.
            #img = cv2.imread('frame0.jpg')
                ret, img = cap.read()
                rows = img.shape[0]
                cols = img.shape[1]
                inp = cv2.resize(img, (300, 300))
                inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
            
                # Run the model
                out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                                sess.graph.get_tensor_by_name('detection_scores:0'),
                                sess.graph.get_tensor_by_name('detection_boxes:0'),
                                sess.graph.get_tensor_by_name('detection_classes:0')],
                               feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
            
                # Visualize detected bounding boxes.
                num_detections = int(out[0][0])
                for i in range(num_detections):
                    classId = int(out[3][0][i])
                    score = float(out[1][0][i])
                    bbox = [float(v) for v in out[2][0][i]]
                    if score > 0.97:
                        x = bbox[1] * cols
                        y = bbox[0] * rows
                        right = bbox[3] * cols
                        bottom = bbox[2] * rows
                        count = count + 1
                        
                        cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
                        if classId == 1:
                            cv2.putText(img, 'Goodday' , (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 255, 51), 1)
                            goodday += 1
                            #postresult_goodday = firebase.patch('/goodday/', {'count': goodday})
                        elif classId == 2:
                            cv2.putText(img, 'BournVita' , (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 255, 51), 1)
                            bournvita +=1
                            ##postresult_bournvita = firebase.patch('/bournvita/', {'count': bournvita})
                        elif classId == 3:
                            cv2.putText(img, 'Gatorade' , (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 255, 51), 1)
                            gatorade +=1
                            #postresult_gatorade = firebase.patch('/gatorade/', {'count': gatorade})
                        else:
                            cv2.putText(img, 'NiceTime', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 255, 51), 1)
                            nicetime += 1
                            #postresult_nicetime = firebase.patch('/nicetime/', {'count': nicetime})
                        #cv2.putText(img, classId[0] , (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                #print("Total : ", count, " Gooday : " , goodday, " Bournvita : ", bournvita, " NiceTime : ", nicetime)
                firebase_god.value = goodday
                firebase_bourn.value,firebase_nice.value = bournvita , nicetime
                firebase_gat.value = gatorade
                cv2.imshow('TensorFlow faster_rcnn', img)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break
        except:
            print("Eception1")

def start(firebase_god,firebase_bourn,firebase_nice,firebase_gat):
    global a,b,c,d
    firebases = firebase.FirebaseApplication(base, None)
    try:
        while True:
            if(a != firebase_god.value or b != firebase_bourn.value or c != firebase_gat.value or d != firebase_nice.value):
                a = {'goodday': firebase_god.value, 'bournvita': firebase_bourn.value, 'gatorade': firebase_gat.value, 'nicetime': firebase_nice.value}
                
                firebases.put('products/2/', 'current_weight', a['goodday'] )
                firebases.put('products/3/', 'current_weight', a['nicetime'] )
                firebases.put('products/4/', 'current_weight', a['gatorade'] )
    except:
        print("Exception2")

    
if __name__ == "__main__":
   firebase_god = Value('f', 0.00)
   firebase_bourn = Value('f', 0.00)
   firebase_nice = Value('f', 0.00)
   firebase_gat = Value('f', 0.00)
   p = Process(target=start, args=(firebase_god,firebase_bourn,firebase_nice,firebase_gat,))
   p.start()
   dothis()
   p.join()

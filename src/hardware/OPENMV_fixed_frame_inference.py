import sensor, image, time, ml, uos, gc

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_NAME = "jordan-ei-ferplus-5-class"  # CHANGE ME

# ============================================================================
# CAMERA INITIALIZATION
# ============================================================================
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((240, 240))
sensor.skip_frames(time=2000) # snapshots to let the camera image stabilize after changing camera settings
sensor.set_auto_gain(False)
sensor.set_auto_exposure(False)

clock = time.clock()

# ============================================================================
# LOAD MODEL AND LABELS
# ============================================================================

model_path = f"/flash/{PROJECT_NAME}/trained.tflite"
net = ml.Model(model_path, load_to_fb=uos.stat(model_path)[6] > (gc.mem_free() - (64*1024)))
labels = [line.rstrip('\n') for line in open(f"/flash/{PROJECT_NAME}/labels.txt")]

print("Model loaded. Classes:", labels)

# ============================================================================
# MAIN LOOP
# ============================================================================
fixed_bbox = (0, 0, 240, 240)
x = fixed_bbox[0]
y = fixed_bbox[1]

while True:
    # Every clock tick, capture an img and crop to fixed bounding box (region of interest)
    clock.tick()
    img = sensor.snapshot()

    # Draw red rectangle around face
    img.draw_rectangle(fixed_bbox, color=(255, 0, 0), thickness=2)

    try:
        # Run prediction
        start_time = time.ticks_ms()
        predictions = net.predict([img])[0].flatten().tolist()
        end_time = time.ticks_ms()
        elapsed = time.ticks_diff(end_time, start_time)
        print(f"Elapsed: {elapsed} ms\tFPS: {1000/elapsed}")

        # Find highest confidence prediction
        max_idx = predictions.index(max(predictions))
        emotion = labels[max_idx]
        confidence = predictions[max_idx] * 100
        print(emotion)

        # Create label text
        label_text = "%s %.0f%%" % (emotion, confidence)

        # Draw black background rectangle for text
        text_width = len(label_text) * 8  # Approximate width
        half_text_width = len(label_text) * 4
        text_height = 10
        img.draw_rectangle(
            120-(half_text_width), 0+text_height,
            text_width + 4, text_height + 4,
            color=(0, 0, 0),
            fill=True
        )

        # Draw text on black background
        img.draw_string(120-(half_text_width), 10, label_text, color=(255, 255, 255))

    except Exception as e:
        print("Classification error:", e)


'''
import sensor, image, time, ml, uos, gc

# Initialize the camera
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((240, 240))
sensor.skip_frames(time=2000)
sensor.set_auto_gain(False)
sensor.set_auto_exposure(False)
clock = time.clock()

# Load face detection cascade
face_cascade = image.HaarCascade("/rom/haarcascade_frontalface.cascade")
net = None
labels = None

#When I plugged in the OpenMV, it showed up as a D:/ drive on my PC.
#So I simply dragged the Edge Impulse folder in there.

project_name = "ei-tinyvision_uploadmodel-openmv-v16" #CHANGE ME

net = ml.Model(f"/flash/{project_name}/trained.tflite", load_to_fb=uos.stat(f'/flash/{project_name}/trained.tflite')[6] > (gc.mem_free() - (64*1024)))
labels = [line.rstrip('\n') for line in open(f"/flash/{project_name}/labels.txt")]


while True:
    clock.tick()
    img = sensor.snapshot()

    faces = img.find_features(face_cascade, threshold=0.5, scale_factor=1.25)

    for face in faces:
        x, y, w, h = face
        img.draw_rectangle(face, color=(255, 0, 0))

        if net is not None:
            try:
                # Clamp ROI
                x2 = max(0, x)
                y2 = max(0, y)
                w2 = min(w, img.width() - x2)
                h2 = min(h, img.height() - y2)

                # Crop face region
                face_img = img.copy(roi=(x2, y2, w2, h2))

                # Run prediction using Edge Impulse format
                # Returns array, flatten and convert to list
                predictions = net.predict([face_img])[0].flatten().tolist()

                # Find highest confidence prediction
                max_idx = 0
                max_conf = predictions[0]
                for i in range(1, len(predictions)):
                    if predictions[i] > max_conf:
                        max_conf = predictions[i]
                        max_idx = i

                emotion = labels[max_idx]
                confidence = max_conf * 100

                # Display result
                img.draw_string(
                    x, y - 15,
                    "%s %.0f%%" % (emotion, confidence),
                    color=(255, 0, 0)
                )

            except Exception as e:
                print("Classification error:", e)
                img.draw_string(x, y - 15, "ERR", color=(255, 0, 0))
        else:
            img.draw_string(x, y - 15, "Face", color=(255, 0, 0))
'''

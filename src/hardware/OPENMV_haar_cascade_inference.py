import sensor, image, time, ml, uos, gc

# ============================================================================
# CONFIGURATION
# When I plugged in the OpenMV, it showed up as a D:/ drive on my PC.
# So I simply dragged the Edge Impulse folder in there.
# ============================================================================
PROJECT_NAME = "ferplus_model_mv_best_converted_model_int8"  # CHANGE ME

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
face_cascade = image.HaarCascade("/rom/haarcascade_frontalface.cascade")

model_path = f"/flash/{PROJECT_NAME}/trained.tflite"
net = ml.Model(model_path, load_to_fb=uos.stat(model_path)[6] > (gc.mem_free() - (64*1024)))
labels = [line.rstrip('\n') for line in open(f"/flash/{PROJECT_NAME}/labels.txt")]

print("Model loaded. Classes:", labels)

# ============================================================================
# MAIN LOOP
# ============================================================================
fixed_bbox = (0, 0, 240, 240)

while True:
    # Every clock tick, capture an img and crop to fixed bounding box (region of interest)
    clock.tick()
    img = sensor.snapshot()
    #roi = img.copy(fixed_bbox)

    # Detect faces
    faces = img.find_features(face_cascade, threshold=0.2, scale_factor=1.4, roi=fixed_bbox)
    img.draw_rectangle(fixed_bbox, color=(0, 255, 0))

    for face in faces:
        x, y, w, h = face

        # Draw red rectangle around face
        img.draw_rectangle(face, color=(255, 0, 0), thickness=2)

        try:
            # Clamp ROI to image bounds
            x2 = max(0, x)
            y2 = max(0, y)
            w2 = min(w, img.width() - x2)
            h2 = min(h, img.height() - y2)

            # Extract face region
            face_img = img.copy(roi=(x2, y2, w2, h2))

            # Run prediction
            start_time = time.ticks_ms()
            predictions = net.predict([face_img])[0].flatten().tolist()
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

            # Calculate text position above face
            text_y = max(10, y - 20)

            # Draw black background rectangle for text
            text_width = len(label_text) * 8  # Approximate width
            text_height = 10
            img.draw_rectangle(
                x - 2, text_y - 2,
                text_width + 4, text_height + 4,
                color=(0, 0, 0),
                fill=True
            )

            # Draw text on black background
            img.draw_string(x, text_y, label_text, color=(255, 255, 255))

        except Exception as e:
            print("Classification error:", e)
            # Draw error message with black background
            img.draw_rectangle(x - 2, y - 22, 30, 14, color=(0, 0, 0), fill=True)
            img.draw_string(x, y - 20, "ERR", color=(255, 0, 0))

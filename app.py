from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import face_recognition
import os
import datetime
import numpy as np
import pandas as pd

import firebase_admin
from firebase_admin import credentials,firestore

cred = credentials.Certificate("employee-monitoring-syst-9c58e-firebase-adminsdk-yvvjq-61fde9871d.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


app = Flask(__name__)

employee_faces_folder = "employee_faces"
if not os.path.exists(employee_faces_folder):
    os.makedirs(employee_faces_folder)

attendance_records = []

def register_employee_face(employee_id):
    video_capture = cv2.VideoCapture(0)
    face_locations = []
    face_encodings = []

    while True:
        ret, frame = video_capture.read(0)

        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Display the frame with face locations marked
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.imshow("Register Face", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save the first captured face encoding
    if face_encodings:
        face_encoding = face_encodings[0]
        np.save(os.path.join(employee_faces_folder, f"employee_{employee_id}"), face_encoding)

    video_capture.release()
    cv2.destroyAllWindows()

    print(f"Employee {employee_id} face registered successfully.")

def mark_attendance():
    video_capture = cv2.VideoCapture(0)
    face_locations = []
    face_encodings = []

    while True:
        ret, frame = video_capture.read()

        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Display the frame with face locations marked
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.imshow("Mark Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Check if any face is recognized           
        if face_encodings:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            employee_id = recognize_employee(face_encodings[0])

            if employee_id:
                existing_record = db.collection('attendance_records').document(employee_id).get().to_dict()

                if existing_record is None or "time_out" in existing_record:
                    attendance_list = existing_record.get("attendance", []) if existing_record else []
                    attendance_list.append({"time_in": current_time, "time_out": None})
                    db.collection('attendance_records').document(employee_id).set({"employee_id": employee_id, "attendance": attendance_list, "in_office": True}, merge=True)
                    print(f"Employee {employee_id} attendance marked with sign-in time at {current_time}")
                else:
                    # Employee has already signed in, update the latest entry
                    latest_entry = attendance_list[-1]
                    if latest_entry["time_out"] is not None:
                        attendance_list.append({"time_in": current_time, "time_out": None})
                        db.collection('attendance_records').document(employee_id).update({"attendance": attendance_list, "in_office": True})
                        print(f"Employee {employee_id} attendance marked with sign-in time at {current_time}")
                    else:
                        print(f"Employee {employee_id} is already signed in.")

                    # Determine in-office status based on the latest entry
                    in_office = latest_entry["time_out"] is None
                    print(f"Employee {employee_id} is {'in' if in_office else 'not in'} the office.")
            else:
                print("Face not recognized. Please try again.")

    video_capture.release()
    cv2.destroyAllWindows()



def recognize_employee(face_encoding):
    for file in os.listdir(employee_faces_folder):
        if file.endswith(".npy"):
            stored_encoding = np.load(os.path.join(employee_faces_folder, file))
            distance = face_recognition.face_distance([stored_encoding], face_encoding)[0]

            # Adjust the threshold as needed
            if distance < 0.5:
                # Extract employee_id from the file name
                return os.path.splitext(file)[0].split("_")[1] 

    return None
def get_in_office_employees():
    in_office_employees = []
    for record in attendance_records:
        if "time_out" not in record:
            in_office_employees.append({"employee_id": record["employee_id"], "time_in": record["time_in"]})
    return in_office_employees

@app.route('/in_office')
def in_office():
    in_office_employees = get_in_office_employees()
    total_in_office = len(in_office_employees)
    return render_template('in_office.html', in_office_employees=in_office_employees, total_in_office=total_in_office)

@app.route('/')
def index():
    return render_template('index.html', attendance_records=attendance_records)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        employee_id = request.form['employee_id']
        register_employee_face(employee_id)
        return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance_route():
    mark_attendance()
    return redirect(url_for('index'))

# @app.route('/download_attendance')
# def download_attendance():
#     csv_data = generate_csv()
#     response = Response(csv_data, content_type='text/csv')
#     response.headers["Content-Disposition"] = "attachment; filename=attendance_records.csv"
#     return response

# def generate_csv():
#     csv_data = "Employee ID,Sign-In Time,Sign-Out Time\n"
#     for record in attendance_records:
#         time_in = record['time_in'].strftime("%Y-%m-%d %H:%M:%S") if 'time_in' in record else ''
#         time_out = record['time_out'].strftime("%Y-%m-%d %H:%M:%S") if 'time_out' in record else ''
#         csv_data += f"{record['employee_id']},{time_in},{time_out}\n"
#     return csv_data





if __name__ == '__main__':
    app.run(debug=True)
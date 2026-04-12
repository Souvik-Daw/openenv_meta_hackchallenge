-- Create doctors table
DROP TABLE IF EXISTS doctors;

CREATE TABLE doctors (
    doctor_id INT ,
    department_id INT NOT NULL,
    day_of_week VARCHAR(10) NOT NULL,
    doctor_name VARCHAR(255) NOT NULL,
    department_name VARCHAR(255) NOT NULL,
    is_available_emergeny BOOLEAN NOT NULL DEFAULT TRUE,
    is_available_opd BOOLEAN NOT NULL DEFAULT TRUE,
    PRIMARY KEY (doctor_id, department_id, day_of_week)
);

-- Optional: index for faster lookups by department or day of week or doctor name
CREATE INDEX idx_doctors_department_id ON doctors(department_id);
CREATE INDEX idx_doctors_day_of_week ON doctors(day_of_week);
CREATE INDEX idx_doctors_doctor_name ON doctors(doctor_name);

-- Insert sample data into doctors table
INSERT INTO doctors (doctor_id, department_id, day_of_week, doctor_name, department_name, is_available_emergeny, is_available_opd) VALUES
(1, 1, 'Monday', 'Dr. Alex Kim', 'Neurology', TRUE, TRUE),
(2, 1, 'Tuesday', 'Dr. Priya Sharma', 'Neurology', TRUE, FALSE),
(3, 1, 'Wednesday', 'Dr. Maria Lopez', 'Neurology', FALSE, TRUE),
(4, 1, 'Thursday', 'Dr. David Chen', 'Neurology', TRUE, TRUE),
(4, 1, 'Friday', 'Dr. David Chen', 'Neurology', TRUE, TRUE),
(2, 1, 'Saturday', 'Dr. Priya Sharma', 'Neurology', TRUE, TRUE),
(1, 1, 'Sunday', 'Dr. Alex Kim', 'Neurology', TRUE, TRUE),

(5, 2, 'Monday', 'Dr. Alex Kim', 'Cardiology', TRUE, TRUE),
(6, 2, 'Tuesday', 'Dr. Samuel Green', 'Cardiology', TRUE, FALSE),
(7, 2, 'Wednesday', 'Dr. Elena Rossi', 'Cardiology', FALSE, TRUE),
(8, 2, 'Thursday', 'Dr. Hannah Park', 'Cardiology', TRUE, TRUE),
(6, 2, 'Friday', 'Dr. Samuel Green', 'Cardiology', TRUE, TRUE),
(7, 2, 'Saturday', 'Dr. Elena Rossi', 'Cardiology', TRUE, TRUE),
(8, 2, 'Sunday', 'Dr. Hannah Park', 'Cardiology', TRUE, TRUE),

(9, 3, 'Monday', 'Dr. Marcus Lee', 'Orthopedics', TRUE, TRUE),
(10, 3, 'Tuesday', 'Dr. Priya Sharma', 'Orthopedics', TRUE, FALSE),
(11, 3, 'Wednesday', 'Dr. Olga Ivanova', 'Orthopedics', FALSE, TRUE),
(9, 3, 'Thursday', 'Dr. Marcus Lee', 'Orthopedics', TRUE, TRUE),
(12, 3, 'Friday', 'Dr. Daniel Moore', 'Orthopedics', TRUE, TRUE),
(10, 3, 'Saturday', 'Dr. Priya Sharma', 'Orthopedics', TRUE, TRUE),
(11, 3, 'Sunday', 'Dr. Olga Ivanova', 'Orthopedics', TRUE, TRUE),

(13, 4, 'Monday', 'Dr. Emily Wong', 'Dental', TRUE, TRUE),
(13, 4, 'Tuesday', 'Dr. Emily Wong', 'Dental', TRUE, TRUE),
(14, 4, 'Wednesday', 'Dr. Raj Patel', 'Dental', TRUE, FALSE),
(15, 4, 'Thursday', 'Dr. Maria Lopez', 'Dental', FALSE, TRUE),
(16, 4, 'Friday', 'Dr. Sophie Turner', 'Dental', TRUE, TRUE),
(14, 4, 'Saturday', 'Dr. Raj Patel', 'Dental', TRUE, TRUE),
(15, 4, 'Sunday', 'Dr. Maria Lopez', 'Dental', TRUE, TRUE),

(17, 5, 'Monday', 'Dr. Omar Khan', 'General Medicine', TRUE, TRUE),
(18, 5, 'Tuesday', 'Dr. Hannah Park', 'General Medicine', TRUE, FALSE),
(19, 5, 'Wednesday', 'Dr. Elena Rossi', 'General Medicine', FALSE, TRUE),
(20, 5, 'Thursday', 'Dr. Daniel Moore', 'General Medicine', TRUE, TRUE),
(17, 5, 'Friday', 'Dr. Omar Khan', 'General Medicine', TRUE, TRUE),
(18, 5, 'Saturday', 'Dr. Hannah Park', 'General Medicine', TRUE, TRUE),
(19, 5, 'Sunday', 'Dr. Elena Rossi', 'General Medicine', TRUE, TRUE),

(21, 6, 'Monday', 'Dr. Samuel Green', 'General Surgery', TRUE, TRUE),
(22, 6, 'Tuesday', 'Dr. Olga Ivanova', 'General Surgery', TRUE, FALSE),
(23, 6, 'Wednesday', 'Dr. Raj Patel', 'General Surgery', FALSE, TRUE),
(21, 6, 'Thursday', 'Dr. Samuel Green', 'General Surgery', TRUE, TRUE),
(24, 6, 'Friday', 'Dr. Emily Wong', 'General Surgery', TRUE, TRUE),
(22, 6, 'Saturday', 'Dr. Olga Ivanova', 'General Surgery', TRUE, TRUE),
(23, 6, 'Sunday', 'Dr. Raj Patel', 'General Surgery', TRUE, TRUE);


-- Create appointments table
DROP TABLE IF EXISTS appointments;

CREATE TABLE appointments (
    appointment_id INT PRIMARY KEY,
    doctor_id INT NOT NULL,
    patient_id INT NOT NULL,
    doctor_name VARCHAR(255) NOT NULL,
    patient_name VARCHAR(255) NOT NULL,
    day DATE NOT NULL,
    CONSTRAINT fk_appointments_doctor FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id)
);

CREATE INDEX idx_appointments_doctor_id ON appointments(doctor_id);
CREATE INDEX idx_appointments_patient_id ON appointments(patient_id);

-- Sample data
INSERT INTO appointments (appointment_id, doctor_id, patient_id, doctor_name, patient_name, day) VALUES
(1, 1, 1001, 'Dr. Alex Kim', 'John Doe', '2026-04-05'),
(2, 5, 1002, 'Dr. Alex Kim', 'Jane Smith', '2026-04-06'),
(3, 9, 1003, 'Dr. Marcus Lee', 'Emily Davis', '2026-04-07'),
(4, 13, 1004, 'Dr. Emily Wong', 'Michael Brown', '2026-04-08');
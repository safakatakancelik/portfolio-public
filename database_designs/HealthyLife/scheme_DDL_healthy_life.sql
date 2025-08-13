DROP SCHEMA public CASCADE;
CREATE SCHEMA public;

CREATE TABLE users (
	id SERIAL PRIMARY KEY,
	created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
	first_name TEXT,
	last_name TEXT,
	dob DATE,
	password_hash TEXT,
	email TEXT -- even this can be taken away? overkill for now
);

CREATE TABLE phone_numbers (
	id SERIAL PRIMARY KEY,
	country_code VARCHAR(5), -- length of these, is it too much of a precaution?
	area_code VARCHAR(7),
	subscriber_number VARCHAR(15)
);

CREATE TABLE phone_number_subscriptions (
	id SERIAL PRIMARY KEY,
	created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
	ended_at TIMESTAMP, -- if null the number is active and associated with the user.
	phone_number_id INT,
	type TEXT, -- home, personal, business
	FOREIGN KEY (phone_number_id) REFERENCES phone_numbers(id)
);

CREATE TABLE users_and_phone_number_subscriptions (
	user_id INT NOT NULL,
	phone_number_subscription_id INT NOT NULL,
	PRIMARY KEY (user_id, phone_number_subscription_id),
	FOREIGN KEY (user_id) REFERENCES users(id),
	FOREIGN KEY (phone_number_subscription_id) REFERENCES phone_number_subscriptions(id)
);


CREATE TABLE patients (
	id SERIAL PRIMARY KEY,
	detail TEXT, --json?
	FOREIGN KEY (id) REFERENCES users(id)
);


CREATE TABLE specialties (
	id SERIAL PRIMARY KEY,
	created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
	name TEXT,
	description TEXT
);


CREATE TABLE doctors (
	id SERIAL PRIMARY KEY,
	detail TEXT, --json?
	specialty_id INT NOT NULL,
	FOREIGN KEY (id) REFERENCES users(id),
	FOREIGN KEY (specialty_id) REFERENCES specialties(id)
);


CREATE TABLE services (
	id SERIAL PRIMARY KEY,
	created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
	name TEXT,
	description TEXT,
	doctor_id INT NOT NULL,
	cost NUMERIC(8,2),
	FOREIGN KEY (doctor_id) REFERENCES doctors(id)

);


CREATE TABLE clinic_rooms (
	id SERIAL PRIMARY KEY,
	description TEXT,
	status TEXT
);

-- Rooms can be used by many doctors, and room assignments can change.
CREATE TABLE doctors_and_clinic_rooms (
	doctor_id INT NOT NULL,
	clinic_room_id INT NOT NULL,
	PRIMARY KEY (doctor_id, clinic_room_id),
	FOREIGN KEY (doctor_id) REFERENCES doctors(id),
	FOREIGN KEY (clinic_room_id) REFERENCES clinic_rooms(id) 
);

CREATE TABLE appointments (
	id SERIAL PRIMARY KEY,
	status VARCHAR(9) CHECK (status in ('scheduled', 'cancelled', 'completed', 'other')),
	description TEXT,
	clinic_room_id INT NOT NULL,
	created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP,
	canceled_at TIMESTAMP,
	start_time TIMESTAMP,
	end_time TIMESTAMP,
	booked_by INT NOT NULL,
	FOREIGN KEY (clinic_room_id) REFERENCES clinic_rooms(id),
	FOREIGN KEY (booked_by) REFERENCES users(id)
);


CREATE TABLE appointments_and_users (
	appointment_id INT NOT NULL,
	user_id INT NOT NULL,
	PRIMARY KEY (appointment_id, user_id),
	FOREIGN KEY (appointment_id) REFERENCES appointments(id),
	FOREIGN KEY (user_id) REFERENCES users(id)
);


CREATE TABLE appointments_and_services (
	appointment_id INT NOT NULL,
	service_id INT NOT NULL,
	PRIMARY KEY (appointment_id, service_id),
	FOREIGN KEY (appointment_id) REFERENCES appointments(id),
	FOREIGN KEY (service_id) REFERENCES services(id)
);
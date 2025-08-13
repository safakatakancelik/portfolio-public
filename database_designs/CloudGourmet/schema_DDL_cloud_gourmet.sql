CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    dob DATE NOT NULL,
    password_hash TEXT NOT NULL,
    email TEXT NOT NULL,
    phone_number TEXT NOT NULL,
    role TEXT NOT NULL -- customer, delivery_agent?, admin?
);

CREATE TABLE customers (
	id INT PRIMARY KEY,
    account_type TEXT NOT NULL, -- individual, family?
    FOREIGN KEY (id) REFERENCES users(id)
);

-- CREATE TABLE families (
--     id SERIAL PRIMARY KEY,
--     name TEXT NOT NULL
-- );
-- CREATE TABLE customers_and_families (
--     customer_id 
-- );

CREATE TABLE delivery_agents (
    id INT PRIMARY KEY,
    FOREIGN KEY (id) REFERENCES users(id)
);


CREATE TABLE addresses (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    street_address TEXT NOT NULL,
    apartment_suite TEXT,
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    postal_code TEXT NOT NULL,
    country TEXT NOT NULL,
    owner_type TEXT NOT NULL,
    owner_id INT NOT NULL,
    FOREIGN KEY (owner_id) REFERENCES users(id)
);

CREATE TABLE delivery_depots (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    address_id INT NOT NULL,
    FOREIGN KEY (address_id) REFERENCES addresses(id)
);

CREATE TABLE delivery_depots_and_delivery_agents (
    delivery_depot_id INT NOT NULL,
    delivery_agent_id INT NOT NULL,
    PRIMARY KEY (delivery_depot_id, delivery_agent_id),
    FOREIGN KEY (delivery_depot_id) REFERENCES delivery_depots(id),
    FOREIGN KEY (delivery_agent_id) REFERENCES delivery_agents(id)
);

CREATE TABLE customers_and_addresses (
    customer_id INT NOT NULL,
    address_id INT NOT NULL,
    PRIMARY KEY (customer_id, address_id),
    FOREIGN KEY (customer_id) REFERENCES customers(id),
    FOREIGN KEY (address_id) REFERENCES addresses(id)
);

CREATE TABLE products(
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    type TEXT NOT NULL
);

CREATE TABLE subscriptions (
    id INT PRIMARY KEY NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP, -- active subscription if null
    customer_id INT NOT NULL, -- subscriber
    -- type TEXT NOT NULL, -- individual, family
    portion_size INT NOT NULL, -- 1 if individual?
    kits_per_week NUMERIC(2, 0) NOT NULL, -- assuming the maximum is 3 x 7 = 21, allowing up to 99
    -- delivery_days NUMERIC(3, 0) NOT NULL, -- base 10 numbers to process in binary terms representing 7 days of the week, "127" aka "1111111" means they prefer at least 1 kit every day
    -- kits_per_days TEXT NOT NULL -- "1234500"
    preferences TEXT NOT NULL, -- JSON,
    remaining_kits_per_week INT NOT NULL,
    FOREIGN KEY (id) REFERENCES products(id)
);


CREATE TABLE subscriptions_and_customers (
    subscription_id INT NOT NULL,
    customer_id INT NOT NULL,
    PRIMARY KEY (subscription_id, customer_id),
    FOREIGN KEY (subscription_id) REFERENCES subscriptions(id),
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);


CREATE TABLE products_and_delivery_depots (
    product_id INT NOT NULL,
    delivery_depot_id INT NOT NULL,
    PRIMARY KEY (product_id, delivery_depot_id),
    FOREIGN KEY (product_id) REFERENCES products(id),
    FOREIGN KEY (delivery_depot_id) REFERENCES delivery_depots(id)
);

CREATE TABLE meal_kits (
    id INT PRIMARY KEY NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    name TEXT NOT NULL,
    sale_price NUMERIC NOT NULL,
    description TEXT NOT NULL,
    type TEXT NOT NULL, -- vegetarian, non-vegetarian, vegan -- can be derived from ingredients as well
    detail TEXT NOT NULL, -- JSON
    FOREIGN KEY (id) REFERENCES products(id)
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    type TEXT NOT NULL,
    status TEXT NOT NULL, -- scheduled, pending, confirmed, in-transit, delivered
    scheduled_at TIMESTAMP,
    order_owner INT NOT NULL, -- customer_id
    order_total NUMERIC NOT NULL,
    discounted_total NUMERIC,
    receipt_link TEXT NOT NULL,
    detail TEXT NOT NULL, -- JSON for example containing discounts applied instead of new table? Order total is the main thing?
    FOREIGN KEY (order_owner) REFERENCES customers(id)
);

CREATE TABLE orders_and_products (
    order_id INT NOT NULL,
    product_id INT NOT NULL,
    PRIMARY KEY (order_id, product_id),
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);


CREATE TABLE suppliers (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    name TEXT NOT NULL,
    specialization TEXT NOT NULL, -- vegan, vegatarian, nonvegan
    detail TEXT NOT NULL -- JSON
);

CREATE TABLE ingredients (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    type TEXT NOT NULL -- vegetarian, non-vegetarian, vegan
);


CREATE TABLE ingredient_supply_contracts (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP, -- null if active
    ingredient_id INT NOT NULL,
    supplier_id INT NOT NULL,
    supplier_price NUMERIC NOT NULL,
    detail TEXT NOT NULL, -- JSON
    FOREIGN KEY (ingredient_id) REFERENCES ingredients(id),
    FOREIGN KEY (supplier_id) REFERENCES suppliers(id)
);



CREATE TABLE meal_kits_and_ingredients (
    meal_kit_id INT NOT NULL,
    ingredient_id INT NOT NULL,
    PRIMARY KEY (meal_kit_id, ingredient_id),
    FOREIGN KEY (meal_kit_id) REFERENCES meal_kits(id),
    FOREIGN KEY (ingredient_id) REFERENCES ingredients(id)
);


CREATE TABLE deliveries (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    status TEXT NOT NULL, -- created, in-transit, delivered, pending
    delivery_agent_id INT NOT NULL,
    order_id INT NOT NULL,
    delivery_depot_id INT NOT NULL,
    delivery_address_id INT NOT NULL,
    detail TEXT NOT NULL, -- JSON
    FOREIGN KEY (delivery_depot_id) REFERENCES delivery_depots(id),
    FOREIGN KEY (delivery_address_id) REFERENCES addresses(id),
    FOREIGN KEY (delivery_agent_id) REFERENCES customers(id),
    FOREIGN KEY (order_id) REFERENCES orders(id)
);

CREATE TABLE discounts (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP, -- null if active
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    discount_type TEXT NOT NULL, -- meal type discount, bulk order discount
    discount_multiplier TEXT NOT NULL -- contains a formula for example n*0.1 where n is the number of items or just a multiplier
);

CREATE TABLE discounts_and_orders (
    discount_id INT NOT NULL,
    order_id INT NOT NULL,
    PRIMARY KEY (discount_id, order_id),
    FOREIGN KEY (discount_id) REFERENCES discounts(id),
    FOREIGN KEY (order_id) REFERENCES orders(id)
);
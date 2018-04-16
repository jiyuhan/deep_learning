batch_size = 100
predict_days = 5

exercise = 5  # Possible values: 1, 2, 3, 4, or 5.

from datasets import generate_x_y_data_v1, generate_x_y_data_v2, generate_x_y_data_v3, generate_x_y_data_v4, generate_x_y_data_v5

# We choose which data function to use below, in function of the exericse.
if exercise == 1:
    generate_x_y_data = generate_x_y_data_v1
if exercise == 2:
    generate_x_y_data = generate_x_y_data_v2
if exercise == 3:
    generate_x_y_data = generate_x_y_data_v3
if exercise == 4:
    generate_x_y_data = generate_x_y_data_v4
if exercise == 5:
    generate_x_y_data = generate_x_y_data_v5

def play_the_game():
    cash = 100000
    shares = 0
    x, y = get_300_day_data()
    print('x is ', x)
    print('y is ', y)


def should_buy(x0, x1, x2, x3, x4, x5):
    if x0 < (x1+x2)/2 < (x3+x4)/2 < x5:
        return True
    else:
        return False


def should_sell(x0, x1, x2, x3, x4, x5):
    if x0 > (x1+x2)/2 > (x3+x4)/2 > x5:
        return True
    else:
        return False


def buy_order(cash, shares, price):
    if cash < 10000:
        return cash, shares  # buy
    buy_share = 10000 // price
    return cash - (buy_share * price), shares + buy_share


def sell_order(cash, shares, price):
    sell_share = shares // 3
    return cash + sell_share * price, shares - sell_share


def get_300_day_data():
    # something something
    return generate_x_y_data(
        isTest=0, batch_size=batch_size, predict_days=predict_days, load_purpose=1)


def predict_five_day():
    # something something
    X, Y = generate_x_y_data(
        isTest=1, batch_size=batch_size, predict_days=predict_days)
    return 1

play_the_game()
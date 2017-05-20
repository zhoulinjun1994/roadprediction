from possion import *
if __name__ == '__main__':
    val_file = '../result/validation_ground.csv'
    #test_file = '../result/linear_qiu_weather_validation.csv'
    #test_file = '../result/naiveknn_val_prediction.csv'
    #test_file = '../result/linear_qiu_validation.csv'
    test_file = '../result/xgboost_weather_validation.csv'
    print mape_travel_time_csv(test_file, val_file)

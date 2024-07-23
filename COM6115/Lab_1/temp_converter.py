def convert_temp(temp):
    fahr = ((temp* 9)/ 5) + 32
    kelv = temp + 273.15
    print("Converting temperature from ", temp, " degrees celsius" )
    print("Temperature in Fahrenheit: ", fahr)
    print("Temperature in Kelvin: ", kelv)


input_user = input("Enter Temperature in Celsius: ")

convert_temp(float(input_user))
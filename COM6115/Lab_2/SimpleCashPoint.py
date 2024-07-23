def cashpoint(truepin,balance):
    if checkPIN(truepin):
        print("\nPlease choose your transaction type")
        print("   - to request a balance  - enter 1")
        print("   - to make a withdrawal  - enter 2")
        print("   - to top-up a telephone - enter 3")
        transactionType = input('\nEnter your choice: ')
        if (transactionType == '1'):
            print("Your balance is", balance, 'GBP')
            print("Thank you for using this service.")
            return "balance-request"
        elif transactionType == '2':
            return withdrawal(balance)
        elif transactionType == '3':
            return mobileTopUp(balance)
        else:
            print("Error: transaction type not recognised.")
            return "transaction-error"
    else:
        print("\nError: PIN authorisation failed - exiting.")
        return "PIN-error"


def checkPIN_once(truepin):
    attempt = input('Please enter your PIN: ')
    if attempt == truepin:
        return True
    else:
        print("PIN incorrect")
        return False

def checkPIN(truepin):
    if checkPIN_once(truepin):
        return True
    print("\nPlease try again (2nd attempt).")
    if checkPIN_once(truepin):
        return True
    print("\nPlease try again (final attempt).")
    if checkPIN_once(truepin):
        return True
    return False

def withdrawal(balance):
    maxWithdrawalAmount = 100.00
    print("Withdrawal amount must be a multiple of 10 pounds.")
    amount = float(input('Please enter your withdrawal amount: '))
    if amount > maxWithdrawalAmount:
        print("Amount exceeds maximum permitted withdrawal amount")
        print("Request refused")
        return "withdrawal-request:refused"
    elif amount > balance:
        print("Amount exceeds available funds")
        print("Request refused")
        return "withdrawal-request:refused"
    elif multpleOfTen(amount):
        print("Please take your cash.")
        print("Your new balance is", balance - amount, 'GBP')
        return ("withdrawal-request",amount)
    else:
        print("Amount is not a multiple of 10")
        print("Request refused")
        return "withdrawal-request:refused"

def multpleOfTen(amount):
    return amount == int(amount / 10.0) * 10

def mobileTopUp(balance):
    maxTopUp = 100.00
    print("Please enter the number of the mobile phone you wish to top-up: ",)
    number1 = input()
    print("Please RE-enter the number of the phone: ",)
    number2 = input()
    if number1 == number2:
        print("Top-up amount must be a multiple of 10 pounds.")
        amount = float(input('Please enter your top-up amount: '))
        if amount > maxTopUp:
            print("Amount exceeds maximum permitted top-up")
            print("Request refused")
            return "top-up-request:refused"
        elif amount > balance:
            print("Amount exceeds available funds")
            print("Request refused")
            return "top-up-request:refused"
        elif multpleOfTen(amount):
            print("Transaction authorised.")
            print("Your new account balance is", balance - amount, 'GBP')
            return ("top-up-request",amount)
        else:
            print("Amount is not a multiple of 10")
            print("Request refused")
            return "top-up-request:refused"
    else:
        print("Error: the two numbers entered are different.")
        print("Transaction cancelled.")
        return "top-up-request:refused"

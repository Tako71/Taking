
SEPARATOR = '------------------------------------------'

# user profile
name = ''
age = 0
phone = ''
email = ''
index = ''
postAdress = ''
information = ''
# enterpreneur links
ogrnip = ''
inn = ''
curentAccount = ''
nameBank = ''
bik = ''
corresAccount = ''



def general_info_users(name_parameter, age_parameter, phone_parameter, email_parameter, index_parameter, postAdress_parameter,infortation_parameter):
    print(SEPARATOR)
    print('Имя:    ', name_parameter)
    if 11 <= age_parameter % 100 <= 19:
        years_parameter = 'лет'
    elif age_parameter % 10 == 1:
        years_parameter = 'год'
    elif 2 <= age_parameter % 10 <= 4:
        years_parameter = 'года'
    else:
        years_parameter = 'лет'

    print('Возраст:', age_parameter, years_parameter)
    print('Телефон:', phone_parameter)
    print('E-mail: ', email_parameter)
    print('Индекс:',index_parameter)
    print('Адрес:',postAdress_parameter)
    if information:
        print('')
        print('Дополнительная информация:')
        print(infortation_parameter)


print('Приложение MyProfile')
print('Сохраняй информацию о себе и выводи ее в разных форматах')

while True:
    # main menu
    print(SEPARATOR)
    print('ГЛАВНОЕ МЕНЮ')
    print('1 - Ввести или обновить информацию')
    print('2 - Вывести информацию')
    print('0 - Завершить работу')

    option = int(input('Введите номер пункта меню: '))
    if option == 0:
        break

    if option == 1:
        # submenu 1: edit info
        while True:
            print(SEPARATOR)
            print('ВВЕСТИ ИЛИ ОБНОВИТЬ ИНФОРМАЦИЮ')
            print('1 - Общая информация')
            print('2 - Информация о предпринимателе')
            print('0 - Назад')

            option2 = int(input('Введите номер пункта меню: '))
            if option2 == 0:
                break
            if option2 == 1:
                # input general info
                name = input('Введите имя: ')
                while 1:
                    # validate user age
                    age = int(input('Введите возраст: '))
                    if age > 0:
                        break
                    print('Возраст должен быть положительным')

                uphone = input('Введите номер телефона (+7ХХХХХХХХХХ): ')
                phone = ''
                for ch in uphone:
                    if ch == '+' or ('0' <= ch <= '9'):
                        phone += ch

                email = input('Введите адрес электронной почты: ')
                index1 = input('Введите почтовый индекс: ')
                index = "".join(c for c in index1 if c.isdecimal())
                postAdress = input('Введите почтовый адрес: ')
                information = input('Введите дополнительную информацию:\n')

            elif option2 == 2:
                # input enterpreneur links
                while 2:
                    ogrnip = input('Введите ОГРНИП: ')
                    if len(ogrnip) != 15:
                        print('ОГРНИП должен состоять из 15 символов!')
                    else:
                        break
                inn = int(input('Введите ИНН: '))
                while 3:
                    curentAccount = input('Введите расчетный счет: ')
                    if len(curentAccount) != 20:
                        print('Расчетный счет должен состоять из 20 символов!')
                    else:
                        break
                nameBank = input('Введите название банка: ')
                bik = int(input('Введите БИК: '))
                corresAccount = int(input('Введите корреспондентский счет: '))
            else:
                print('Введите корректный пункт меню')
    elif option == 2:
        # submenu 2: print info
        while True:
            print(SEPARATOR)
            print('ВЫВЕСТИ ИНФОРМАЦИЮ')
            print('1 - Общая информация')
            print('2 - Вся информация')
            print('0 - Назад')

            option2 = int(input('Введите номер пункта меню: '))
            if option2 == 0:
                break
            if option2 == 1:
                general_info_users(name, age, phone, email, index, postAdress, information)

            elif option2 == 2:
                general_info_users(name, age, phone, email, index, postAdress, information)

                # print social links
                print('')
                print('Информация о предпринимателе')
                print('ОГРНИП:', ogrnip )
                print('ИНН: ', inn)
                print('Р/c:   ', curentAccount )
                print('Название банка: ', nameBank )
                print('БИК: ', bik )
                print('К/c: ', corresAccount )



            else:
                print('Введите корректный пункт меню')
    else:
        print('Введите корректный пункт меню')
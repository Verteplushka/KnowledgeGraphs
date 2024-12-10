from owlready2 import *
import xml.etree.ElementTree as ET

size_mapping = {
    'T': 'Крошечный',
    'S': 'Маленький',
    'M': 'Средний',
    'L': 'Большой',
    'H': 'Огромный',
    'G': 'Громадный'
}

type_mapping = {
    'гуманоид (ааракокра), monster manual': 'Гуманоид',
    'искаженное, monster manual': 'Абберация',
    'чудовище, monster manual': 'Монстр',
    'гуманоид (любая раса), monster manual': 'Гуманоид',
    'дракон, monster manual': 'Дракон',
    'нежить, monster manual': 'Нежить',
    'элементаль, monster manual': 'Элементаль',
    'зверь, monster manual': 'Зверь',
    'механизм, monster manual': 'Конструкт',
    'исчадие (юголот), monster manual': 'Исчадие',
    'растение, monster manual': 'Растение',
    'исчадие (демон), monster manual': 'Исчадие',
    'исчадие (дьявол), monster manual': 'Исчадие',
    'слизь, monster manual': 'Слизь',
    'фея, monster manual': 'Фея',
    'гуманоид (гоблиноид), monster manual': 'Гуманоид',
    'гуманоид (балливаг), monster manual': 'Гуманоид',
    'исчадие, monster manual': 'Исчадие',
    'исчадье (дьявол), monster manual': 'Исчадие',
    'исчадье (демон), monster manual': 'Исчадие',
    'искаженный, monster manual': 'Абберация',
    'конструкция, monster manual': 'Конструкт',
    'великан, monster manual': 'Великан',
    'небожитель, monster manual': 'Небожитель',
    'Элементаль, monster manual': 'Элементаль',
    'искаженный (меняющий форму), monster manual': 'Абберация',
    'гуманоид (гном), monster manual': 'Гуманоид',
    'божественный, monster manual': 'Небожитель',
    'чудовище (меняющий форму), monster manual': 'Монстр',
    'гуманоид (эльф), monster manual': 'Гуманоид',
    'гуманоид (дворф), monster manual': 'Гуманоид',
    'божественный (титан), monster manual': 'Небожитель',
    'giant, monster manual': 'Великан',
    'гуманоид (гит), monster manual': 'Гуманоид',
    'гуманоид (гнолл), monster manual': 'Гуманоид',
    'гуманоид (гримлок), monster manual': 'Гуманоид',
    'гуманоид (человек), monster manual': 'Гуманоид',
    'демон (дьявол), monster manual': 'Исчадие',
    'гуманоид (меняющий форму), monster manual': 'Гуманоид',
    'гуманоид (кенку), monster manual': 'Гуманоид',
    'гуманоид (кобольд), monster manual': 'Гуманоид',
    'чудовище (титан), monster manual': 'Монстр',
    'гуманоид (куо-тоа), monster manual': 'Гуманоид',
    'гуманоид (ящерочеловек), monster manual': 'Гуманоид',
    'гуманоид (мерфолк), monster manual': 'Гуманоид',
    'гуманоид (орк), monster manual': 'Гуманоид',
    'гуманоид (куаггот), monster manual': 'Гуманоид',
    'гуманоид (сахуагин), monster manual': 'Гуманоид',
    'исчадие (меняющий форму), monster manual': 'Исчадие',
    'рой Крошечных зверей, monster manual': 'Стая_крошечных_зверей',
    'гуманоид (три-крин), monster manual': 'Гуманоид',
    'гуманоид (троглодит), monster manual': 'Гуманоид',
    'гигант, monster manual': 'Великан',
    'нежить (меняющий форму), monster manual': 'Нежить',
    'гуманоид (человек, меняющий форму), monster manual': 'Гуманоид',
    'чудовище (меняющий форму, юоан-ти), monster manual': 'Монстр',
    'гуманоид (юоан-ти), monster manual': 'Гуманоид',
    'мертвяк, monster manual': 'Нежить'
}

alignment_mapping = {
    'нейтрально-доброе': 'Нейтральный_добрый',
    'законно-злое': 'Законопослушный_злой',
    'хаотично-злое': 'Хаотичный_злой',
    'любое мировоззрение': 'Истинно_нейтральный',
    'хаотично-доброе': 'Хаотичный_добрый',
    'законно-доброе': 'Законопослушный_добрый',
    'нейтральное': 'Истинно_нейтральный',
    'без мировоззрения': 'Истинно_нейтральный',
    'законно-нейтральное': 'Законопослушный_нейтральный',
    'нейтрально-злое': 'Нейтральный_злой',
    'любое не доброе': 'Нейтральный_злой',
    'любое не законное': 'Истинно_нейтральный',
    'любое хаотическое': 'Хаотичный_нейтральный',
    'хаотично-нейтральное': 'Хаотичный_нейтральный',
    'любое злое': 'Законопослушный_злой',
    'unaligned': 'Истинно_нейтральный',
    'neutral good': 'Нейтральный_добрый',
    'lawful evil': 'Законопослушный_злой',
    'chaotic evil': 'Хаотичный_злой',
    'chaotic neutral': 'Хаотичный_нейтральный',
    'neutral good (50%) or neutral evil (50%)': 'Нейтральный_добрый',
    'any alignment': 'Истинно_нейтральный',
    'chaotic good': 'Хаотичный_добрый',
    'lawful good': 'Законопослушный_добрый',
    'neutral evil': 'Нейтральный_злой',
    'any non-good alignment': 'Нейтральный_злой',
    'нейтрально злой': 'Нейтральный_злой',
    'хаотично-доброе (75%) или нейтрально-злое (25%)': 'Хаотичный_добрый',
    'хаотично-злой': 'Хаотичный_злой',
}

# onto = get_ontology("Ontology_dnd_test.owl").load()
#
# onto_classes_names = []
# for cls in onto.classes():
#     onto_classes_names.append(cls.name)
#
# print(onto_classes_names)
#
# with onto:
#     if "Enemy" in onto_classes_names:
#         Enemy = onto.Enemy
#     else:
#         print("Класс Enemy не найден, создайте его сначала.")
#         exit()
#
#     goblin = Enemy("Goblin")
#
# onto.save(file="Ontology_dnd_test.owl", format="rdfxml")



def normalize_data(data):
    enemies = []
    for enemy in data:
        enemy['size'] = size_mapping.get(enemy['size'])
        enemy['type'] = type_mapping.get(enemy['type'])
        enemy['alignment'] = alignment_mapping.get(enemy['alignment'])
        pattern = r"^\d+"
        enemy['ac'] = re.match(pattern, enemy['ac']).group(0)
        enemy['hp'] = re.match(pattern, enemy['hp']).group(0)
        if enemy['resist'] not in enemies:
            enemies.append(enemy['resist'])

    print(data)
    print(enemies)


def xml_to_dict(element):
    result = {}
    name_element = element.find('name')
    if name_element is not None and name_element.text:
        result['name'] = name_element.text.strip()
    attributes = [
        'size', 'type', 'alignment', 'ac', 'hp', 'str', 'dex', 'con',
        'int', 'wis', 'cha', 'resist', 'vulnerable', 'immune',
        'conditionImmune', 'languages', 'cr'
    ]
    for attribute in attributes:
        element_attribute = element.find(attribute)
        if element_attribute is not None and element_attribute.text:
            result[attribute] = element_attribute.text.strip()
        else:
            result[attribute] = None
    return result


def parse_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        result = []
        for monster in root.findall('monster'):
            result.append(xml_to_dict(monster))
        return result
    except ET.ParseError as e:
        print(f"Ошибка парсинга XML: {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


file_path = 'monsters.xml'
data = parse_xml(file_path)
normalize_data(data)


from owlready2 import *
import xml.etree.ElementTree as ET
import random

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

damage_type_mapping = {
    'некротика': ['Некротический'],
    'молния; гром; дробящий, колющий, и рубящий от немагического оружия': ['Электрический', 'Дробящий', 'Колющий', 'Рубящий'],
    'холод, огонь, молния; дробящий, колющий, и рубящий от немагического оружия': ['Холодный', 'Огненный', 'Электрический', 'Дробящий', 'Колющий', 'Рубящий'],
    'урон от заклинаний; не магический дробящий, колющий, и рубящий (от Каменной Кожи)': ['Импровизированный_урон', 'Дробящий', 'Колющий', 'Рубящий'],
    'яд': ['Ядовитый'],
    'колющий': ['Колющий'],
    'дробящий, колющий': ['Дробящий', 'Колющий'],
    'холод, молния; дробящий, колющий, и рубящий от немагического оружия': ['Холодный', 'Электрический', 'Дробящий', 'Колющий', 'Рубящий'],
    'кислота, огонь, молния, гром; дробящий, колющий, и рубящий от немагического оружия': ['Кислотный', 'Огненный', 'Электрический', 'Дробящий', 'Колющий', 'Рубящий'],
    'холод; дробящий, колющий, и рубящий от немагического оружия кроме посеребренного': ['Холодный', 'Дробящий', 'Колющий', 'Рубящий'],
    'холод, огонь, молния': ['Холодный', 'Огненный', 'Электрический'],
    'кислота, холод, огонь, молния': ['Кислотный', 'Холодный', 'Огненный', 'Электрический'],
    'холод, огонь, молния, яд; дробящий, колющий, и рубящий от немагического оружия': ['Холодный', 'Огненный', 'Электрический', 'Ядовитый', 'Дробящий', 'Колющий', 'Рубящий'],
    'холод; дробящий, колющий и рубящий - если оружие не магическое (или серебрянное)': ['Холодный', 'Дробящий', 'Колющий', 'Рубящий'],
    'холод, огонь, электричество': ['Холодный', 'Огненный', 'Электрический'],
    'radiant': ['Излучение'],
    'дробящий, колющий, и рубящий от магического оружия': ['Дробящий', 'Колющий', 'Рубящий'],
    'излучение; дробящий, колющий, и рубящий от немагического оружия': ['Излучение', 'Дробящий', 'Колющий', 'Рубящий'],
    'огонь': ['Огненный'],
    'дробящий, колющий, и рубящий от немагического оружия': ['Дробящий', 'Колющий', 'Рубящий'],
    'молния, некротика, колющий': ['Электрический', 'Некротический', 'Колющий'],
    "bludgeoning, piercing, and slashing from nonmagical weapons that aren't adamantine": ['Дробящий', 'Колющий', 'Рубящий'],
    'кислота, холод, огонь': ['Кислотный', 'Холодный', 'Огненный'],
    'дробящий, колющий, and рубящего урона от немагического оружия': ['Дробящий', 'Колющий', 'Рубящий'],
    'дробящий, колющий, и рубящий от немагического оружия кроме адамантиевого': ['Дробящий', 'Колющий', 'Рубящий'],
    'дробящий, колющий и рубящий немагическим оружием, которое не посеребрено': ['Дробящий', 'Колющий', 'Рубящий'],
    'холод; дробящий, колющий, и рубящий from nonmagical/nonsilver weapons': ['Холодный', 'Дробящий', 'Колющий', 'Рубящий'],
    'холод': ['Холодный'],
    'холод, молния, некротика': ['Холодный', 'Электрический', 'Некротический'],
    'кислота, холод, молния': ['Кислотный', 'Холодный', 'Электрический'],
    'холод, огонь, электричество; дробящий, колющий и рубящий (от немагического оружия)': ['Холодный', 'Огненный', 'Электрический', 'Дробящий', 'Колющий', 'Рубящий'],
    'холод, огонь; дробящий, колющий, и рубящий от немагического оружия кроме посеребренного': ['Холодный', 'Огненный', 'Дробящий', 'Колющий', 'Рубящий'],
    'кислота': ['Кислотный'],
    'кислота, холод, огонь, молния, гром; дробящий, колющий, и рубящий от немагического оружия': ['Кислотный', 'Холодный', 'Огненный', 'Электрический', 'Дробящий', 'Колющий', 'Рубящий'],
    'холод; огонь; молния; дробящий, колющий, и рубящий от немагического оружия': ['Холодный', 'Огненный', 'Электрический', 'Дробящий', 'Колющий', 'Рубящий'],
    'некротика, психика': ['Некротический', 'Психическая_энергия'],
    'кислота, холод, огонь, молния, звук; дробящий, колющий, и рубящий урон от немагического оружия': ['Кислотный', 'Холодный', 'Огненный', 'Электрический', 'Звуковой', 'Дробящий', 'Колющий', 'Рубящий'],
    'кислота, огонь, некротика, гром; дробящий, колющий, и рубящий от немагического оружия': ['Кислотный', 'Огненный', 'Некротический', 'Дробящий', 'Колющий', 'Рубящий'],
    'холод, огонь': ['Холодный', 'Огненный'],
    'дробящий, колющий, рубящий': ['Дробящий', 'Колющий', 'Рубящий'],
    'некротика; дробящий, колющий, и рубящий от немагического оружия': ['Некротический', 'Дробящий', 'Колющий', 'Рубящий'],
    'кислота; дробящий, колющий, и рубящий от немагического оружия': ['Кислотный', 'Дробящий', 'Колющий', 'Рубящий'],
    'огонь; дробящий, колющий, и рубящий от немагического оружия': ['Огненный', 'Дробящий', 'Колющий', 'Рубящий'],
    'кислота, холод, огонь, некротика, гром; дробящий, колющий, и рубящий от немагического оружия': ['Кислотный', 'Холодный', 'Огненный', 'Некротический', 'Дробящий', 'Колющий', 'Рубящий'],
    'кислота, холод, огонь, молния, гром; дробящий, колющий, и рубящий от немагического оружия кроме посеребренного': ['Кислотный', 'Холодный', 'Огненный', 'Электрический', 'Дробящий', 'Колющий', 'Рубящий'],
    'колющий и рубящий от немагического и неадамантиевого оружия': ['Колющий', 'Рубящий'],
    'гром': ['Звуковой'],
    'психика': ['Психическая_энергия'],
    'дробящий, огонь': ['Дробящий', 'Огненный'],
    'дробящий': ['Дробящий'],
    'колющий от магического оружия wielded by good creatures': ['Колющий'],
    'излучение': ['Излучение'],
    'necrotic, яд': ['Некротический', 'Ядовитый'],
    'огонь, яд': ['Огненный', 'Ядовитый'],
    'холод, некротика, яд': ['Холодный', 'Некротический', 'Ядовитый'],
    'кислота, холод, молния, рубящий': ['Кислотный', 'Холодный', 'Электрический', 'Рубящий'],
    'кислота, яд, психическая энергия; дробящий, колющий и рубящий (не в случае магического или адамантинового оружия)': ['Кислотный', 'Ядовитый', 'Психическая_энергия', 'Дробящий', 'Колющий', 'Рубящий'],
    'психическая энергия; дробящий, колющий, рубящий (немагическое оружие)': ['Психическая_энергия', 'Дробящий', 'Колющий', 'Рубящий'],
    'некротика, яд': ['Некротический', 'Ядовитый'],
    'некротика, яд, психика; дробящий, колющий, и рубящий от немагического оружия': ['Некротический', 'Ядовитый', 'Психическая_энергия', 'Дробящий', 'Колющий', 'Рубящий'],
    'молния, гром': ['Электрический'],    'холод, огонь, яд': ['Холодный', 'Огненный', 'Ядовитый'],
    'молния, яд; дробящий, колющий, и рубящий от немагического оружия кроме адамантиевого': ['Электрический', 'Ядовитый', 'Дробящий', 'Колющий', 'Рубящий'],    'force, некротика, яд': ['Силовое_поле', 'Некротический', 'Ядовитый'],
    'холод, яд': ['Холодный', 'Ядовитый'],
    'огонь, яд, психика; дробящий, колющий, и рубящий от немагического оружия кроме адамантиевого': ['Огненный', 'Ядовитый', 'Психическая_энергия', 'Дробящий', 'Колющий', 'Рубящий'],
    'дробящий, колющий и рубящий урон немагическим оружием, которое не посеребрено': ['Дробящий', 'Колющий', 'Рубящий'],
    'молния; дробящий, колющий, и рубящий от немагического оружия': ['Электрический', 'Дробящий', 'Колющий', 'Рубящий'],
    'яд; дробящий, колющий, и рубящий от немагического оружия': ['Ядовитый', 'Дробящий', 'Колющий', 'Рубящий'],
    'некротика, яд; дробящий, колющий, и рубящий от немагического оружия': ['Некротический', 'Ядовитый', 'Дробящий', 'Колющий', 'Рубящий'],
    'молния, рубящий': ['Электрический', 'Рубящий'],
    'холод, молния, яд': ['Холодный', 'Электрический', 'Ядовитый'],
    "яд, психика; дробящий, колющий, и рубящий от немагического оружия that aren't adamantine": ['Ядовитый', 'Психическая_энергия', 'Дробящий', 'Колющий', 'Рубящий'],
    'огонь, яд; дробящий, колющий, и рубящий от немагического оружия': ['Огненный', 'Ядовитый', 'Дробящий', 'Колющий', 'Рубящий'],
    'дробящий, колющий, and рубящего урона от немагического оружия кроме посеребренного': ['Дробящий', 'Колющий', 'Рубящий'],
    'дробящий, колющий, и рубящий урон от немагического оружия кроме посеребренного': ['Дробящий', 'Колющий', 'Рубящий'],
    'некротика; дробящий, колющий, и рубящий от немагического оружия кроме посеребренного': ['Некротический', 'Дробящий', 'Колющий', 'Рубящий']
}

condition_immune_mapping = {
    'очарование, усталость, испуг, паралич, отравление': ['Очарованный', 'Истощенный', 'Испуганный', 'Парализованный', 'Отравленный'],
    'отравление': ['Отравленный'],
    'истощение, захват, паралич, окаменение, отравление, сбивание с ног, удерживание, лишение сознания': [
        'Истощенный', 'Схваченный', 'Парализованный', 'Окаменевший', 'Отравленный', 'Сбитый_с_ног', 'Схваченный', 'Бессознательный'
    ],
    'очарование, испуг': ['Очарованный', 'Испуганный'],
    'ослепление, очарование, оглушение, истощение, испуг, паралич, окаменение, отравление': [
        'Ослеплённый', 'Очарованный', 'Оглохший', 'Истощенный', 'Испуганный', 'Парализованный', 'Окаменевший', 'Отравленный'
    ],
    'очарование, отравление': ['Очарованный', 'Отравленный'],
    'очарование, испуг, паралич, окаменение, отравление, лишение сознания': [
        'Очарованный', 'Испуганный', 'Парализованный', 'Окаменевший', 'Отравленный', 'Бессознательный'
    ],
    'очарование, истощение, испуг, захват, паралич, окаменение, отравление, сбивание с ног, удерживание': [
        'Очарованный', 'Истощенный', 'Испуганный', 'Схваченный', 'Парализованный', 'Окаменевший', 'Отравленный', 'Сбитый_с_ног', 'Схваченный'
    ],
    'сбивание с ног': ['Сбитый_с_ног'],
    'ослепление, очарование, оглушение, истощение, испуг, сбивание с ног': [
        'Ослеплённый', 'Очарованный', 'Оглохший', 'Истощенный', 'Испуганный', 'Сбитый_с_ног'
    ],
    'очарование, истощение, паралич, отравление': ['Очарованный', 'Истощенный', 'Парализованный', 'Отравленный'],
    'очарование, .усталость, испуг, паралич, окаменение, отравление': [
        'Очарованный', 'Истощенный', 'Испуганный', 'Парализованный', 'Окаменевший', 'Отравленный'
    ],
    'окаменение': ['Окаменевший'],
    'истощение, испуг, отравление': ['Истощенный', 'Испуганный', 'Отравленный'],
    'очарование, истощение, паралич, окаменение, отравление, сбивание с ног': [
        'Очарованный', 'Истощенный', 'Парализованный', 'Окаменевший', 'Отравленный', 'Сбитый_с_ног'
    ],
    'очарование, оглушение, истощение, испуг, паралич, окаменение, отравление, сбивание с ног, ошеломление': [
        'Очарованный', 'Оглохший', 'Истощенный', 'Испуганный', 'Парализованный', 'Окаменевший', 'Отравленный', 'Сбитый_с_ног', 'Ошеломлённый'
    ],
    'очарование, истощение, испуг': ['Очарованный', 'Истощенный', 'Испуганный'],
    'истощение, паралич, окаменение, отравление, лишение сознания': [
        'Истощенный', 'Парализованный', 'Окаменевший', 'Отравленный', 'Бессознательный'
    ],
    'очарование, испуг, паралич, отравление': ['Очарованный', 'Испуганный', 'Парализованный', 'Отравленный'],
    'очарование, истощение, испуг, паралич, окаменение, отравление': [
        'Очарованный', 'Истощенный', 'Испуганный', 'Парализованный', 'Окаменевший', 'Отравленный'
    ],
    'ослепление, очарование, оглушение, испуг, паралич, окаменение, отравление': [
        'Ослеплённый', 'Очарованный', 'Оглохший', 'Испуганный', 'Парализованный', 'Окаменевший', 'Отравленный'
    ],
    'истощение, паралич, отравление, окаменение': ['Истощенный', 'Парализованный', 'Отравленный', 'Окаменевший'],
    'истощение, окаменение, отравление': ['Истощенный', 'Окаменевший', 'Отравленный'],
    'ослепление, оглушение, испуг, паралич, отравление, сбивание с ног': [
        'Ослеплённый', 'Оглохший', 'Испуганный', 'Парализованный', 'Отравленный', 'Сбитый_с_ног'
    ],
    'ослепление, сбивание с ног': ['Ослеплённый', 'Сбитый_с_ног'],
    'ослепление, очарование, оглушение, испуг, паралич, окаменение, отравление, ошеломление': [
        'Ослеплённый', 'Очарованный', 'Оглохший', 'Испуганный', 'Парализованный', 'Окаменевший', 'Отравленный', 'Ошеломлённый'
    ],
    'ослепление': ['Ослеплённый'],
    'испуг, паралич': ['Испуганный', 'Парализованный'],
    'очарование, испуг, отравление': ['Очарованный', 'Испуганный', 'Отравленный'],
    'очарование, истощение, испуг, паралич, отравление': [
        'Очарованный', 'Истощенный', 'Испуганный', 'Парализованный', 'Отравленный'
    ],
    'испуг': ['Испуганный'],
    'истощение, отравление': ['Истощенный', 'Отравленный'],
    'некротика, отравление': ['Отравленный'],
    'ослепление, оглушение': ['Ослеплённый', 'Оглохший'],
    'очарование': ['Очарованный'],
    'очарование, истощение, захват, паралич, окаменение, отравление, сбивание с ног, удерживание, лишение сознания': [
        'Очарованный', 'Истощенный', 'Схваченный', 'Парализованный', 'Окаменевший', 'Отравленный', 'Сбитый_с_ног', 'Схваченный', 'Бессознательный'
    ],
    'ослепление, очарование, испуг, паралич, отравление': [
        'Ослеплённый', 'Очарованный', 'Испуганный', 'Парализованный', 'Отравленный'
    ],
    'очарование, истощение, испуг, паралич, отравление, ошеломление': [
        'Очарованный', 'Истощенный', 'Испуганный', 'Парализованный', 'Отравленный', 'Ошеломлённый'
    ],
    'очарование, истощение, испуг, паралич, отравление, лишение сознания': [
        'Очарованный', 'Истощенный', 'Испуганный', 'Парализованный', 'Отравленный', 'Бессознательный'
    ],
    'истощение, испуганный, схваченный, парализованный, окаменение, отравление, сбит с ног, удерживаемый': [
        'Истощенный', 'Испуганный', 'Схваченный', 'Парализованный', 'Окаменевший', 'Отравленный', 'Сбитый_с_ног', 'Схваченный'
    ],
    'истощение, захват, паралич, окаменение, отравление, сбивание с ног, удерживание': [
        'Истощенный', 'Схваченный', 'Парализованный', 'Окаменевший', 'Отравленный', 'Сбитый_с_ног', 'Схваченный'
    ],
    'ослепление, оглушение, истощение': ['Ослеплённый', 'Оглохший', 'Истощенный'],
    'ослепление, оглушение, испуг': ['Ослеплённый', 'Оглохший', 'Испуганный'],
    'очарование, истощение, испуг, отравление': ['Очарованный', 'Истощенный', 'Испуганный', 'Отравленный'],
    'очарование, испуг, паралич, окаменение, сбивание с ног, удерживание, ошеломление': [
        'Очарованный', 'Испуганный', 'Парализованный', 'Окаменевший', 'Сбитый_с_ног', 'Схваченный', 'Ошеломлённый'
    ],
    'очарование, паралич, отравление': ['Очарованный', 'Парализованный', 'Отравленный'],
    'истощение, захват, паралич, отравление, удерживание, сбивание с ног, лишение сознания': [
        'Истощенный', 'Схваченный', 'Парализованный', 'Отравленный', 'Сбитый_с_ног', 'Схваченный', 'Бессознательный'
    ],
    'poisoned': ['Отравленный']
}

language_mapping = {
    'Воздушный': ['Первичный'],
    'Глубинная речь, телепатия 24 клетки': ['Глубинная_Речь'],    'один любой язык (обычно Общий)': ['Общий'],  # Предполагается, что чаще всего Общий.    'Огненный': ['Первичный'],  # Первичный включает Огненный.
    'Бездны, телепатия 24 клеток': ['Бездны'],    'Общий, Эльфийский': ['Общий', 'Эльфийский'],
    'Инфернальный, телепатия 24 клеток': ['Инфернальный'],
    'Драконий': ['Драконий'],
    'Глубинная Речь, Подземный': ['Глубинная_Речь', 'Подземный'],
    'понимает Глубинную Речь и Подземный но не может разговаривать': ['Глубинная_Речь', 'Подземный'],
    'понимает Sylvan но не может разговаривать': ['Сильван'],    'Инфернальный, телепатия 24 клетки': ['Инфернальный'],
    'Общий плюс любой другой язык': ['Общий'],  # Поскольку это один язык (Общий) + любой другой язык.
    'Общий, Гоблинский': ['Общий', 'Гоблинский'],    'Бездны, Общий, Инфернальный': ['Бездны', 'Общий', 'Инфернальный'],
    'Эльфийский, Сильван': ['Эльфийский', 'Сильван'],
    'инфернальный, телепатия 24 клетки': ['Инфернальный'],
    'Бездны, телепатия 24 клетки': ['Бездны'],
    'понимает драконий, но не может на нем говорить': ['Драконий'],
    'понимает глубинную речь (но не умеет разговаривать)': ['Глубинная_Речь'],    'Глубинный язык, Подземный': ['Глубинная_Речь', 'Подземный'],
    'Общий, Великаний': ['Общий', 'Великаний'],
    'любой язык (как правило Общий)': ['Общий'],    'понимает Общий, но не умеет на нем разговаривать': ['Общий'],  # Понимание Общего языка.
    'любой один язык (обычно Общий)': ['Общий'],
    'Великаний': ['Великаний'],    'Бездны, Общий': ['Бездны', 'Общий'],    'Гномий, Земляной, Подземный': ['Гномий', 'Подземный'],    'Общий': ['Общий'],
    'Водный, Драконий': ['Драконий'],
    'Бездны, телепатия 12 клетки (только с существами, понимающими язык Бездны)': ['Бездны'],
    'Эльфийский, Подземный': ['Эльфийский', 'Подземный'],
    'Druidic плюс любые 2 языка': ['Друидический_язык'],    'Великаний, Орочий': ['Великаний', 'Орочий'],    'понимает Подземный но не может разговаривать, телепатия 12 клеток': ['Подземный'],    'Гигантских орлов, понимает Общий и Воздушный но не может разговаривать': ['Общий'],  # Воздушный не входит в итоговый список.
    'Гигантских лосей, понимает Общий, Эльфийский, и Сильван но не может разговаривать': ['Общий', 'Эльфийский', 'Сильван'],
    'Гигантских сов понимает Общий, Эльфийский и Сильван, но не может разговаривать': ['Общий', 'Эльфийский', 'Сильван'],    'Бездны, Гнолл': ['Бездны', 'Гнолл'],
    'Гнолл': ['Гнолл'],
    'Бездны': ['Бездны'],
    'Общий, Драконий, Сильван': ['Общий', 'Драконий', 'Сильван'],    'Подземный': ['Подземный'],
    'Небесный, Общий': ['Небесный', 'Общий'],
    'Общий, Сфинксов': ['Общий'],  # Сфинксов не входит в итоговый список.
    'понимает Инфернальный но не может разговаривать': ['Инфернальный'],    'Инферальный, телепатия 24 клетки': ['Инфернальный'],  # Инферальный считается как Инфернальный.    'Инфернальный, Общий': ['Инфернальный', 'Общий'],
    'понимает Глубинная Речь но не может разговаривать, телепатия 12 клеток': ['Глубинная_Речь'],
    'Воздушный, понимает Общий но не может разговаривать': ['Общий'],
    'Общий (не может говорить в облике шакала)': ['Общий'],
    'понимает Воздушный и Общий но может говорить только с помощью свойства Мимикрия': ['Общий'],
    'понимает Бездны, Небесный, Инфернальный и Изначальный но не может разговаривать, телепатия 24 клетки': ['Бездны', 'Небесный', 'Инфернальный'],
    'понимает адский но не может разговаривать': ['Инфернальный'],  # Адский считается как Инфернальный.
    'Общий плюс до 5 других языков': ['Общий'],  # Мы учитываем только "Общий", остальные языки не указаны.    'понимает Бездны но не может разговаривать': ['Бездны'],    'Водный, Общий': ['Общий'],
    'Бездны, Водный': ['Бездны'],
    'Бездны, Инфернальный, телепатия 12 клетки': ['Бездны', 'Инфернальный'],    'понимает Общий (но не говорит на нем)': ['Общий'],
    'Бездны, Общий, Инфернальный, Изначальный': ['Бездны', 'Общий', 'Инфернальный'],
    'понимает Бездны, Общий, и Инфернальный, но не может разговаривать': ['Бездны', 'Общий', 'Инфернальный'],
    'Бездны, Инфернальный, телепатия 12 клетки': ['Бездны', 'Инфернальный'],
    'понимает Общий and Великаний но не может разговаривать': ['Общий', 'Великаний'],
    'Общий, Orc': ['Общий', 'Орочий'],    'понимает Небесный, Общий, Эльфийский и Сильван, но не может разговаривать': ['Общий', 'Эльфийский', 'Сильван'],
    'понимает Общий and Эльфийский но не может разговаривать': ['Общий', 'Эльфийский'],    'Общий (не может говорить в форме медведя)': ['Общий'],  # Язык не изменяется.
    'Общий (не может говорить в форме кабана)': ['Общий'],  # Язык не изменяется.
    'Общий (не может разговаривать в форме крысы)': ['Общий'],  # Язык остается общий, только форма не позволяет говорить.
    'Общий (не может разговаривать в форме тигра)': ['Общий'],  # Язык остается общий, только форма не позволяет говорить.
    'Общий (не может разговаривать в форме волка)': ['Общий'],  # Язык остается общий, только форма не позволяет говорить.
    'Общий, Великаний, Зимних волков': ['Общий', 'Великаний'],  # Зимние волки не входят в итоговый список, но есть Общий и Великаний.
    'Гоблинский, Воргов': ['Гоблинский', 'Воровской_жаргон'],  # Языки Гоблинский и Воровской жаргон.
    'Бездны, Эльфийский, Подземный': ['Бездны', 'Эльфийский', 'Подземный'],  # Все эти языки входят в итоговый список.
    'Бездны, Общий, Драконий': ['Бездны', 'Общий', 'Драконий'],  # Все эти языки входят в итоговый список.
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
    locations = ['Болото', 'Город', 'Горы', 'Деревня', 'Лес', 'Побережье', 'Под_водой', 'Подземелья', 'Подземье',
         'Полярная_тундра', 'Пустыня', 'Равнина/луг', 'Руины', 'Тропики', 'Холмы']
    random.seed(52)
    for enemy in data:
        enemy['size'] = size_mapping.get(enemy['size'])
        enemy['type'] = type_mapping.get(enemy['type'])
        enemy['alignment'] = alignment_mapping.get(enemy['alignment'])
        pattern = r"^\d+"
        enemy['ac'] = re.match(pattern, enemy['ac']).group(0)
        enemy['hp'] = re.match(pattern, enemy['hp']).group(0)
        enemy['resist'] = damage_type_mapping.get(enemy['resist'])
        enemy['vulnerable'] = damage_type_mapping.get(enemy['vulnerable'])
        enemy['immune'] = damage_type_mapping.get(enemy['immune'])
        enemy['conditionImmune'] = condition_immune_mapping.get(enemy['conditionImmune'])
        enemy['languages'] = language_mapping.get(enemy['languages'])
        enemy['location'] = random.choice(locations)
        if enemy['immune'] not in enemies:
            enemies.append(enemy['immune'])

    print(data)
    print(enemies)
    print(len(enemies))


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


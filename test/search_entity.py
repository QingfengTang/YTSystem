import pandas as pd
import sys
import os
sys.path.append('..')
from src.utils import config

def search_entity():
    path = os.path.join(config.BASE_DIR, 'data\\raw_data', 'entity_property.xlsx')
    e = pd.read_excel(path)
    e = e.where(e.notnull(), None)
    entity_properties = {}
    for i in range(e.columns.size):
        property = e.columns[i]
        entity_properties[i] = property
    entity_names = {}
    entity_contents = {}

    for i in range(e.values.shape[0]):
        values_i = e.loc[i].values
        properties = {}
        for j in range(values_i.size):
            value = values_i[j]
            properties[entity_properties[j]] = value
            if entity_properties[j] == '实体名称':
                name = value
                name_value = [name]
            elif entity_properties[j] == '中文名' and value is not None:
                # 将value按|划分，name_value.append(value)
                chinese_names = value.split('|')
                for chinese_name in chinese_names:
                    name_value.append(chinese_name)

            elif entity_properties[j] == '外文名' and value is not None:
                # 将value按|划分，name_value.append(value)
                english_names = value.split("|")
                for english_name in english_names:
                    name_value.append(english_name)

            elif entity_properties[j] == '别名' and value is not None:
                # 将value按|划分，name_value.append(value)
                other_names = value.split("|")
                for other_name in other_names:
                    name_value.append(other_name)

        entity_names[i] = set(name_value)
        entity_contents[i] = properties

    search_name = input("请输入你要查询的人名:")
    for i , value_names in entity_names.items():
        if search_name in value_names:
            print(entity_contents[i])
            break
    if(i==len(entity_names)-1):
        print("没有该实体属性信息！")

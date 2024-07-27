from scipy.optimize import linprog
import numpy as np
import pandas as pd
import os
import itertools
from collections import OrderedDict
from time import time
pd.set_option('mode.chained_assignment', None)

params = {
    "run_allocation_shortage": False,
    # data_capacity_someplants 1.4.1 - hardcopy.xlsx #data_capacity_700plants 1.3.6 - hardcopy.xlsx
    "input_file": "input\input_data_700plants 1.3.7 - HC - advantage.xlsx",
    "input_file_distance1": "input\Copy of plants dis 021109 - HC1.xlsx",
    "input_file_distance2": "input\Copy of plants dis 021109 - HC2.xlsx",
    "input_file_advantage_carrying": "input/advantage_carrying 1.0.3.xlsx",
    "distance_sheet": "Plant dis. cal",  # plants dis.
    "plant_sheet": "INPUT",  # plants
    "output_file_name": "allocation_shortage_output1.0.0.xlsx",
    "code_name_sheet": "code_ostan",
    "drop_duplicate": False

}
shortage_columns = [
    "factory_name",
    "factory_code",
    "state",
    "year",
    "product",
    "code",
    "capacity",
    "supply",
    "receive",
    "send",
    "shortage",
    "surplus"
]

shortage_columns_persian = [
    "کارخانه",
    "کد کارخانه",
    "وضعیت",
    "سال",
    "محصول",
    "جایگاه",
    "ظرفیت",
    "تولید",
    "دریافت",
    "ارسال",
    "کمبود",
    "مازاد"
]

allocation_columns = [
    "origin_name",
    "origin",
    "destination_name",
    "destination",
    "state",
    "year",
    "product",
    "code",
    "allocation",
]

allocation_columns_persian = [
    "مبدا",
    "کد مبدا",
    "مقصد",
    "کد مقصد",
    "وضعیت",
    "سال",
    "محصول",
    "جایگاه",
    "تخصیص"
]

change_dict = {
    10: "سنگ آهن",
    20: "کنستانتره",
    30: "گندله",
    40: "آهن اسفنجی",
    51: "اسلب",
    52: "بیلت",
    61: "ورق گرم",
    72: "ورق سرد",
    71: "لوله",
    81: "سایر محصولاتِ تخت",
    62: "میلگرد",
    63: "سایرِ مقاطعِ بلند",
}

start_year = 1401
finish_year = 1402
# year_list = [year for year in range(start_year, finish_year+1)]
year_list = [1401, 1402, 1403, 1404, 1405, 1410]
# state_list = ["فعال"]  # "احتمال متوسط", "احتمال کم", "مازاد"
state_list = ["فعال", "احتمال بالا", "احتمال متوسط", "احتمال کم", "مازاد"]

def create_required_folders():
    specific_path = '' 
    folders = ['input', 'output']
    for folder in folders:
        folder_path = os.path.join(specific_path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

def find_duplicate(dataframe):
    # find duplicate edges in the input data frame
    if len(dataframe) != len(dataframe.drop_duplicates()):
        print("there is duplicated in dataframe")
        duplicates = dataframe[dataframe.duplicated(keep=False)]
        duplicates = (duplicates.groupby(duplicates.columns.tolist())
                      .apply(lambda x: tuple(x.index))
                      .reset_index(name='idx'))
        print(duplicates)


def load_distance_plants(distance_file_name):
    plants_dis = pd.read_excel(
        params[f"{distance_file_name}"], sheet_name=params["distance_sheet"])
    plants_dis = plants_dis.drop(0)
    return plants_dis

def making_distance_melt_df(plants_dis):
    plants_dis = plants_dis.drop(plants_dis.columns[1:3], axis=1)
    plants_dis.index = list(plants_dis["factory_province_code"])
    plants_dis.drop(columns={"factory_province_code"}, inplace=True)
    plants_dis_df = plants_dis.reset_index().melt(id_vars='index', var_name='Column Index', value_name='Value')
    plants_dis_df.rename(columns = {"index":"plant_sender", 'Column Index':"plant_receiver", "Value":"distance"}, inplace=True)
    return plants_dis_df


def load_adv_carrying_data():
    adv_carrying_data = pd.read_excel(params["input_file_advantage_carrying"])
    return adv_carrying_data


def making_distance_matrix(plants_dis):
    plants_dis = plants_dis.drop(plants_dis.columns[:3], axis=1)
    distance_matix = plants_dis.to_numpy()
    return distance_matix


def load_code_name_province():
    code_name_data = pd.read_excel(
        params["input_file"], sheet_name=params["code_name_sheet"], header=0)
    # need_columns = plants_dis[["province", "province_code"]]
    return code_name_data


def making_coefficients_vector(plants_dis, sender, receiver):
    plants_dis = plants_dis[["factory_province_code"]+list(receiver)]
    plants_dis = plants_dis[plants_dis["factory_province_code"].isin(sender)]
    plants_dis.drop(columns=["factory_province_code"], axis=1, inplace=True)
    coefficients_vector = plants_dis.to_numpy().flatten()
    return coefficients_vector


def sender_receiver_plants(sender, receiver):
    sender_plants = np.repeat(sender, len(receiver))
    df = pd.DataFrame({k: receiver
                      for k in sender})
    receiver_plants = pd.Series(df.values.ravel('F'))
    return sender_plants, receiver_plants


def making_some_matrix(number_row, number_column):
    zero_matrix = np.zeros(
        (number_row, number_column),  dtype=np.int8)
    identity_matrix = np.matrix(np.identity(
        number_column), copy=False,  dtype=np.int8)
    counter_list = [i for i in range(number_row)]
    return zero_matrix, identity_matrix, counter_list


def making_constraints_matrix_part1(zero_matrix, row_number):
    sub_matrix = zero_matrix.copy()
    sub_matrix[row_number, :] = 1
    return sub_matrix


def making_constraints_matrix(number_row, number_column, zero_matrix, identity_matrix, counter_list):
    return np.concatenate((np.concatenate(list(map(lambda x: making_constraints_matrix_part1(zero_matrix, x), counter_list)), axis=1),
                           np.concatenate([identity_matrix for counter in range(
                               len(counter_list))], axis=1), np.array([[-1 for counter in range(number_row*number_column)]])), axis=0)


def making_bound(number_row, number_column, sender, receiver, supply, demand):
    bound = [(0, float('inf'))
             for counter in range(number_row*number_column)]
    for item in sender:
        if item in receiver:
            s_index = sender.index(item)
            r_index = receiver.index(item)
            position = s_index*number_column + r_index
            bound[position] = (
                min(supply[s_index], demand[r_index]), float('inf'))
    return bound


def create_steel_chain():
    steel_chain = pd.DataFrame([[10, 20], [20, 30], [30, 40], [40, 51], [40, 52], [51, 61], [
                               52, 62], [52, 63], [61, 71], [61, 72], [72, 81]], columns=["feed", "product"])
    return steel_chain


def load_supply_data(state, source_supply):
    # source_supply = pd.read_excel(params["input_file"], sheet_name="plants") #sheet_name=params["plant_sheet"])
    source_supply = source_supply[source_supply["state"] == state]
    find_duplicate(source_supply[["category_code", "factory_province_code"]])
    source_supply.drop_duplicates(
        subset=["category_code", "factory_province_code"], inplace=True)
    return source_supply


def create_supply_data_per_year(source_supply, plants_dis, year):
    supply_data = pd.DataFrame()
    supply_data["factory_code"] = plants_dis["factory_province_code"]
    # supply_data["factory_name"] = plants_dis["factory_name"]
    for item in steel_chain_items:
        supply_data[item] = supply_data["factory_code"].map(
            source_supply[source_supply["category_code"] == item].set_index("factory_province_code")[year])
    supply_data = supply_data.fillna(0)
    return supply_data
# supply_data = create_supply_data(state_list[0], 1401)


def reset_dataframe(state, year, sender_plants, receiver_plants, plants_dis):
    shortage = pd.DataFrame()
    allocation = pd.DataFrame()
    shortage["factory_code"] = plants_dis["factory_province_code"]
    shortage["year"] = year
    shortage["state"] = state
    allocation["origin"] = sender_plants
    allocation["destination"] = receiver_plants
    allocation["year"] = year
    allocation["state"] = state
    return allocation, shortage


# (shortage, product, supply, capacity, receive, send, feed, demand)
def making_shortage(shortage, feed, capacity, receiver_list, sender_list, optimization_supply_df, supply, demand):

    receiver_df = pd.DataFrame(list(zip(receiver_list, list(
        optimization_supply_df.sum()))), columns=["code", "receiver"])
    sender_df = pd.DataFrame(list(zip(sender_list, list(
        optimization_supply_df.sum(axis=1)))), columns=["code", "sender"])
    supply_df = pd.DataFrame(
        list(zip(sender_list, supply)), columns=["code", "supply"])
    demand_df = pd.DataFrame(
        list(zip(receiver_list, demand)), columns=["code", "demand"])

    shortage["capacity"] = capacity
    shortage["product"] = feed
    shortage["code"] = feed
    if feed == 10:
        shortage["supply"] = 0
    else:
        shortage["supply"] = shortage["factory_code"].map(
            supply_df.set_index("code")["supply"])
    shortage["receive"] = shortage["factory_code"].map(
        receiver_df.set_index("code")["receiver"])
    shortage["send"] = shortage["factory_code"].map(
        sender_df.set_index("code")["sender"])
    shortage["demand"] = shortage["factory_code"].map(
        demand_df.set_index("code")["demand"])
    shortage[["supply", "demand", "receive", "send"]] = shortage[[
        "supply", "demand", "receive", "send"]].fillna(0)
    shortage["shortage"] = [
        max(0, val) for val in (shortage["demand"] - shortage["receive"])]
    shortage["surplus"] = [max(0, val) for val in (
        shortage["capacity"]-shortage["send"])]
    return shortage


def making_allocation(allocation, product, allocation_value, feed):
    allocation["product"] = product
    allocation["allocation"] = allocation_value
    allocation["code"] = feed
    return allocation


def minof_two_list(counter, list1, list2):
    return min(list1[counter], list2[counter])


def distribute_optimum_between_products(optimization_supply_df, supply_chain_optimal, product_list, product):
    optimization_supply_df_copy = optimization_supply_df.copy()
    for column_code in optimization_supply_df_copy.columns:
        sub_df = supply_chain_optimal[supply_chain_optimal["factory_code"] == column_code]
        sum_supply = (sub_df.filter(product_list).sum(
            axis='columns')).values[0]
        if sum_supply != 0:
            optimization_supply_df_copy[column_code] *= sub_df[product].values[0]/sum_supply
        else:
            optimization_supply_df_copy[column_code] = 0
    return optimization_supply_df_copy


def optimal_transfer(supply_data, state, year):
    start_time = time()
    allocation_list = []
    shortage_list = []
    optimal_dict = {}
    sender_list_dict = {}
    # [10, 20, 30, 40, 51, 61, 71, 72, 81, 52, 62, 63]
    for feed in steel_chain_items:
        # print(feed)
        product_list = list(
            steel_chain[steel_chain["feed"] == feed]["product"])
        if len(product_list) != 0:
            non_zero_demand = supply_data[(
                supply_data[product_list] != 0).any(axis=1)]
            receiver_list = list(non_zero_demand["factory_code"])
            if (feed == steel_chain["feed"][0]):
                non_zero_supply = supply_data[supply_data[feed] != 0]
                supply = list(non_zero_supply[feed])
                sender_list = list(non_zero_supply["factory_code"])
            else:
                feed_code = steel_chain[steel_chain["product"]
                                        == feed]["feed"].values[0]
                supply = [item for item in optimal_dict[f"{feed_code}{feed}_optimal"] if item != 0]
                sender_list = sender_list_dict[feed]
            sender_plants, receiver_plants = sender_receiver_plants(
                sender_list, receiver_list)
            allocation, shortage = reset_dataframe(
                state, year, sender_plants, receiver_plants, plants_dis)
            demand = list(non_zero_demand[product_list].sum(axis='columns'))
            number_row = len(sender_list)
            number_column = len(receiver_list)
            # print([number_row, number_column])
            bound = making_bound(number_row, number_column,
                                 sender_list, receiver_list, supply, demand)
            right_side_constraints = supply + demand + \
                [-min([sum(supply), sum(demand)])]
            start_time = time()
            zero_matrix, identity_matrix, counter_list = making_some_matrix(
                number_row, number_column)
            optimization_supply = linprog(making_coefficients_vector(plants_dis, sender_list, receiver_list),
                                          making_constraints_matrix(
                                              number_row, number_column, zero_matrix, identity_matrix, counter_list),
                                          right_side_constraints, bounds=bound, options={
                'cholesky': False, 'sym_pos': False})
            # print(f"finish time is {time()-start_time}")
            optimization_supply_df = pd.DataFrame(optimization_supply["x"].reshape(
                number_row, number_column), columns=receiver_list)
            if (product_list[0] == 81):
                allocation = reset_dataframe(
                    state, year, sender_plants, receiver_plants, plants_dis)[0]
                allocation = making_allocation(
                    allocation, product_list[0], pd.Series(optimization_supply_df.values.ravel('C')), product_list[0])
                allocation_list.append(allocation)
            if len(product_list) > 1:
                for product in product_list:
                    optimization_supply_df_copy = distribute_optimum_between_products(
                        optimization_supply_df, supply_data, product_list, product)
                    optimal_dict[f"{feed}{product}_optimal"] = list(
                        optimization_supply_df_copy.sum())
                    sender_list_dict[product] = [
                        column for column in optimization_supply_df_copy.columns if optimization_supply_df_copy[column].sum() != 0]
                    if (product in list(steel_chain[~steel_chain["product"].isin(steel_chain["feed"])]["product"])):
                        allocation = reset_dataframe(
                            state, year, sender_plants, receiver_plants, plants_dis)[0]
                        allocation = making_allocation(
                            allocation, product, pd.Series(optimization_supply_df_copy.values.ravel('C')), product)
                        allocation_list.append(allocation)
            else:
                optimal_dict[f"{feed}{product_list[0]}_optimal"] = list(
                    optimization_supply_df.sum())
                sender_list_dict[product_list[0]] = [
                    column for column in optimization_supply_df.columns if optimization_supply_df[column].sum() != 0]
            shortage = making_shortage(
                shortage, feed, supply_data[feed], receiver_list, sender_list, optimization_supply_df, supply, demand)
            shortage_list.append(shortage)
            allocation = reset_dataframe(
                state, year, sender_plants, receiver_plants, plants_dis)[0]
            allocation = making_allocation(
                allocation, feed, optimization_supply["x"], feed)
            allocation_list.append(allocation)
        else:
            shortage = reset_dataframe(
                state, year, sender_plants, receiver_plants, plants_dis)[1]
            feed_code = steel_chain[steel_chain["product"]
                                    == feed]["feed"].values[0]
            shortage = making_shortage(
                shortage, feed, supply_data[feed], [
                ],  sender_list_dict[feed], pd.DataFrame(),
                [item for item in optimal_dict[f"{feed_code}{feed}_optimal"] if item != 0], [])
            shortage_list.append(shortage)
    allocation = pd.concat(allocation_list, axis=0)
    shortage = pd.concat(shortage_list, axis=0)
    # print(f"finish time is {time()-start_time}")
    return allocation, shortage


def concat_dataframes(list_dataframe, len_list):
    df1 = pd.concat([list_dataframe[i][0] for i in range(len_list)], axis=0)
    df2 = pd.concat([list_dataframe[i][1] for i in range(len_list)], axis=0)
    return df1, df2


def remove_zero_allocation_and_reset_index(allocation, shortage):
    # allocation = allocation[~allocation["allocation"].isin([0])]
    allocation = allocation[allocation["allocation"] > 1]
    # shortage = shortage[~((shortage["capacity"]==0)&(shortage["supply"]==0)&(shortage["send"]==0)\
    #                      &(shortage["receive"]==0)&(shortage["demand"]==0)&(shortage["shortage"]==0))]
    allocation.reset_index(inplace=True, drop=True)
    shortage.reset_index(inplace=True, drop=True)
    return allocation, shortage


def output_feed(supply_data, state, year):
    # output = list(map(lambda feed: optimal_transfer(supply_chain_optimal, feed, state, year), [10, 20, 30, 40, 51, 61, 71, 72, 81, 52, 62, 63]))
    return optimal_transfer(supply_data, state, year)


def output_year(all_supply_data, state, plants_dis):
    def functions_for_year(year):
        print(year)
        supply_data = create_supply_data_per_year(
            all_supply_data, plants_dis, year)
        return output_feed(supply_data, state, year)
    output = list(map(lambda year: functions_for_year(year), year_list))
    return concat_dataframes(output, len(year_list))


def output_state(plants_dis):
    def functions_for_state(state):
        # print(state)
        all_supply_data = load_supply_data(state, source_supply)
        return output_year(all_supply_data, state, plants_dis)
    output = list(map(lambda state: functions_for_state(state), state_list))
    return concat_dataframes(output, len(state_list))


def rename_string_output(allocation, shortage):
    shortage = shortage[shortage_columns]
    allocation = allocation[allocation_columns]
    shortage[["state", "product"]] = shortage[[
        "state", "product"]].replace(change_dict)
    allocation[["state", "product"]] = allocation[[
        "state", "product"]].replace(change_dict)
    shortage.columns = shortage_columns_persian
    allocation.columns = allocation_columns_persian
    return allocation, shortage


def change_code_to_name(allocation, shortage, code_name_data):

    code_name_province = code_name_data[[
        "factory_province_code", "factory_name"]].drop_duplicates()

    allocation["origin_name"] = allocation["origin"].map(
        code_name_province.set_index("factory_province_code")["factory_name"])
    allocation["destination_name"] = allocation["destination"].map(
        code_name_province.set_index("factory_province_code")["factory_name"])
    shortage["factory_name"] = shortage["factory_code"].map(
        code_name_province.set_index("factory_province_code")["factory_name"])
    shortage = shortage[shortage_columns]
    allocation = allocation[allocation_columns]
    shortage[["state", "product"]] = shortage[[
        "state", "product"]].replace(change_dict)
    allocation[["state", "product"]] = allocation[[
        "state", "product"]].replace(change_dict)
    shortage.columns = shortage_columns_persian
    allocation.columns = allocation_columns_persian

    return allocation, shortage


def validation_results(allocation, shortage):

    # try:
    #     allocation = pd.read_excel("balanced capacity plants finalla.xlsx", sheet_name="allocation")
    #     valid_allocation = pd.read_excel("validation/balanced capacity plants final_province.xlsx", sheet_name="allocation")
    #     for feed in list(OrderedDict.fromkeys(list(steel_chain["feed"])+list(steel_chain["product"]))):
    #         valid_allocation_feed = valid_allocation[valid_allocation["جایگاه"]==feed]
    #         valid_allocation_feed.reset_index(inplace=True, drop=True)
    #         allocation_feed = allocation[allocation["جایگاه"]==feed]
    #         allocation_feed.reset_index(inplace=True, drop=True)
    #         compare_output = pd.DataFrame(valid_allocation_feed.compare(
    #                     allocation_feed, align_axis=1, keep_shape=False, keep_equal=False))
    #         if len(compare_output) != 0:
    #             compare_output.to_excel(f"compare_output_allocation_{feed}.xlsx")
    #             print(
    #                 f"we have difference in allocation for {feed}:\n, {compare_output}")
    # except Exception as e:
    #     print("Can only compare identically-labeled DataFrame objects", f"{feed}_allocation")

    # try:
    #     shortage = pd.read_excel("balanced capacity plants finalla.xlsx", sheet_name="shortage")
    #     valid_shortage = pd.read_excel("validation/balanced capacity plants final_province.xlsx", sheet_name="shortage")
    #     for feed in list(OrderedDict.fromkeys(list(steel_chain["feed"])+list(steel_chain["product"]))):
    #         valid_shortage_feed = valid_shortage[valid_shortage["جایگاه"]==feed]
    #         valid_shortage_feed.reset_index(inplace=True, drop=True)
    #         shortage_feed = shortage[shortage["جایگاه"]==feed]
    #         shortage_feed.reset_index(inplace=True, drop=True)
    #         compare_output = pd.DataFrame(valid_shortage_feed.compare(
    #             shortage_feed, align_axis=1, keep_shape=False, keep_equal=False))
    #         if len(compare_output) != 0:
    #             compare_output.to_excel(f"compare_output_shortage_{feed}.xlsx")
    #             print(
    #                 f"we have difference in shortage for {feed}:\n, {compare_output}")
    # except Exception as e:
    #     print("Can only compare identically-labeled DataFrame objects", f"{feed}_shortage")

    # valid_allocation = pd.read_excel("validation/balanced capacity cities finalmain.xlsx", sheet_name="allocation")
    # for feed in list(OrderedDict.fromkeys(list(steel_chain["feed"])+list(steel_chain["product"]))):
    #     valid_allocation_feed = valid_allocation[valid_allocation["جایگاه"]==feed]
    #     valid_allocation_feed.reset_index(inplace=True, drop=True)
    #     allocation_feed = allocation[allocation["جایگاه"]==feed]
    #     allocation_feed.reset_index(inplace=True, drop=True)
    #     compare_output = pd.DataFrame(valid_allocation_feed.compare(
    #                 allocation_feed, align_axis=1, keep_shape=False, keep_equal=False))
    #     if len(compare_output) != 0:
    #         compare_output.to_excel(f"compare_output_allocation_{feed}.xlsx")
    #         print(
    #             f"we have difference in allocation for {feed}:\n, {compare_output}")
    # valid_shortage = pd.read_excel("validation/balanced capacity cities finalmain.xlsx", sheet_name="shortage")
    # for feed in list(OrderedDict.fromkeys(list(steel_chain["feed"])+list(steel_chain["product"]))):
    #     valid_shortage_feed = valid_shortage[valid_shortage["جایگاه"]==feed]
    #     valid_shortage_feed.reset_index(inplace=True, drop=True)
    #     shortage_feed = shortage[shortage["جایگاه"]==feed]
    #     shortage_feed.reset_index(inplace=True, drop=True)
    #     compare_output = pd.DataFrame(valid_shortage_feed.compare(
    #         shortage_feed, align_axis=1, keep_shape=False, keep_equal=False))
    #     if len(compare_output) != 0:
    #         compare_output.to_excel(f"compare_output_shortage_{feed}.xlsx")
    #         print(
    #             f"we have difference in shortage for {feed}:\n, {compare_output}")

    valid_allocation = pd.read_excel(
        "validation/balanced capacity plants_checking1.1.1.xlsx", sheet_name="allocation")
    try:
        pd.testing.assert_frame_equal(allocation.reset_index(drop=True), valid_allocation.reset_index(
            drop=True), check_dtype=False)  # rtol=1e-15, atol=1e-15
    except Exception as e:
        compare_output = pd.DataFrame(allocation.reset_index(drop=True).compare(
            valid_allocation.reset_index(drop=True), align_axis=1, keep_shape=False, keep_equal=False))
        if len(compare_output) != 0:
            print(
                f"Failed test! Difference in values for allocation", compare_output)

    valid_shortage = pd.read_excel(
        "validation/balanced capacity plants_checking1.1.1.xlsx", sheet_name="shortage")
    try:
        pd.testing.assert_frame_equal(shortage.reset_index(drop=True), valid_shortage.reset_index(
            drop=True), check_dtype=False)  # rtol=1e-15, atol=1e-15
    except Exception as e:
        compare_output = pd.DataFrame(shortage.reset_index(drop=True).compare(
            valid_shortage.reset_index(drop=True), align_axis=1, keep_shape=False, keep_equal=False))
        if len(compare_output) != 0:
            print(
                f"Failed test! Difference in values for shortage", compare_output)

    # valid_allocation = pd.read_excel("validation/balanced capacity plants_checking1.1.1.xlsx", sheet_name="allocation")
    # compare_output = pd.DataFrame(valid_allocation.compare(allocation, align_axis=1, keep_shape=False, keep_equal=False))
    # if len(compare_output) != 0:
    #     compare_output.to_excel(f"compare_output_allocation.xlsx")
    #     print(
    #         f"we have difference in allocation:\n, {compare_output}")
    # valid_shortage = pd.read_excel("validation/balanced capacity plants_checking1.1.1.xlsx", sheet_name="shortage")
    # compare_output = pd.DataFrame(valid_shortage.compare(
    #     shortage, align_axis=1, keep_shape=False, keep_equal=False))
    # if len(compare_output) != 0:
    #     compare_output.to_excel(f"compare_output_shortage.xlsx")
    #     print(
    #         f"we have difference in shortage:\n, {compare_output}")


def get_excel(input1, input2):
    graph_dict = {'allocation': input1, 'shortage': input2}
    with pd.ExcelWriter('output/allocation_shortage_output.xlsx') as writer:
        for sheetname, graphname in graph_dict.items():
            graphname.to_excel(writer, sheet_name=sheetname, index=False)


def calculate_adv_carrying_values(distance_matrix, standard_capacity, avelable_initial_material, transferring_cost):
    num_cities = len(standard_capacity)
    adv_carrying_values = np.zeros(num_cities)
    for i in range(num_cities):
        required_initial_material = standard_capacity[i]
        if required_initial_material == 0:
            continue
        city_distances = [(j, distance_matrix[i][j]) for j in range(
            num_cities) if avelable_initial_material[j] > 0]
        city_distances.sort(key=lambda x: x[1])
        total_adv_carrying_value = 0
        for city_idx, dist in city_distances:
            if required_initial_material <= 0:
                break
            available = avelable_initial_material[city_idx]
            initial_material_to_take = min(
                required_initial_material, available)
            total_adv_carrying_value += transferring_cost * dist * initial_material_to_take
            required_initial_material -= initial_material_to_take
        adv_carrying_values[i] = total_adv_carrying_value
    return adv_carrying_values


if __name__ == "__main__":
    start_time = time()
    create_required_folders()
    plants_dis = load_distance_plants("input_file_distance1")
    plants_dis_df = making_distance_melt_df(plants_dis)
    steel_chain = create_steel_chain()
    shortage = pd.read_excel(f'output/{params["output_file_name"]}', sheet_name="shortage")
    allocation = pd.read_excel(f'output/{params["output_file_name"]}', sheet_name="allocation")
    vertical_yeild = pd.read_excel(params["input_file"], sheet_name="vertical yeild")
    plants_dis_df.rename(columns={"plant_sender":"کد مبدا", "plant_receiver":"کد مقصد"}, inplace=True)
    allocation = pd.merge(allocation, plants_dis_df, how="left", on = ["کد مبدا", "کد مقصد"],)
    allocation["عدد حمل"] = 0.02*allocation["تخصیص"]*allocation["distance"]
    allocation.drop(columns=["distance"], inplace=True)
    for code in set(list(steel_chain["feed"])+list(steel_chain["product"])):
        ver_yeild = vertical_yeild[vertical_yeild["دسته"]==code]["بازده"].values[0]
        allocation.loc[allocation["جایگاه"]==code, "عدد حمل"] /= ver_yeild
    sub_allo = allocation[["کد مقصد", "وضعیت", "سال", "جایگاه", "عدد حمل"]]
    sub_allo = sub_allo.groupby(["کد مقصد", "وضعیت", "سال", "جایگاه"]).agg({"عدد حمل":"sum"}).reset_index()
    sub_allo.rename(columns={"کد مقصد":"کد کارخانه"}, inplace=True)
    shortage2 = pd.merge(shortage, sub_allo, how="left", on=["کد کارخانه", "وضعیت", "سال", "جایگاه"])
    shortage2["عدد حمل"]=shortage2["عدد حمل"].fillna(0)
    get_excel(allocation, shortage2)
    if params["run_allocation_shortage"]:
        code_name_data = load_code_name_province()
        source_supply = pd.read_excel(
            params["input_file"], sheet_name=params["plant_sheet"])
        #steel_chain = create_steel_chain()
        steel_chain_items = list(OrderedDict.fromkeys(
            list(steel_chain["feed"])+list(steel_chain["product"])))
        allocation, shortage = output_state(plants_dis)
        allocation, shortage = remove_zero_allocation_and_reset_index(
            allocation, shortage)
        # allocation, shortage = rename_string_output(allocation, shortage)
        allocation, shortage = change_code_to_name(
            allocation, shortage, code_name_data)
        get_excel(allocation, shortage)
        print(f"finish time is {time()-start_time}")
        validation_results(allocation, shortage)
    ###############################################################
    plants_dis = load_distance_plants("input_file_distance2")
    distance_matrix = making_distance_matrix(plants_dis)
    adv_carrying_data = load_adv_carrying_data()
    vertical_yeild = pd.read_excel(params["input_file"], sheet_name="vertical yeild")
    if not params["run_allocation_shortage"]:
        shortage = pd.read_excel(f'output/{params["output_file_name"]}', sheet_name="shortage")
    #product_list = [20, 30, 40, 51, 52, 61, 72, 62, 63, 71, 81]
    #transferring_cost = list(adv_carrying_data["transferring_cost"].unique())
    total_adv_carrying_values = []
    total_year_list = []
    subsub_adv_carrying_total = pd.DataFrame()
    #number_list = [500000, 1000000, 1700000, 2000000, 2500000, 3000000, 3400000]
    #for i in range(len(product_list)):
    #counter = 0
    for state in adv_carrying_data["status"].unique(): # ["فعال"]
        print(state)
        for year in [1405]: #year_list
            for product in steel_chain["product"]: #  [20, 30, 40, 51, 52, 61, 62, 72, 81]
                feed = steel_chain[steel_chain["product"]==product]["feed"].unique()[0]
                sub_surplus_shortage = shortage[(shortage["وضعیت"]==state) & \
                                            (shortage["سال"]==year) & (shortage["جایگاه"]==feed)]
                sub_surplus_shortage = sub_surplus_shortage[sub_surplus_shortage["مازاد"] > 1]
                sub_adv_carrying = adv_carrying_data[(adv_carrying_data["status"]==state) 
                                                        & (adv_carrying_data["product_code"]==product)]
                transferring_cost = sub_adv_carrying["transferring_cost"].unique()[0]
                vert_yeild = vertical_yeild[vertical_yeild["دسته"]==product]["بازده"].values[0]
                vert_yeild_feed = vertical_yeild[vertical_yeild["دسته"]==feed]["بازده"].values[0]
                for number in sub_adv_carrying["capacity"].unique():
                    subsub_adv_carrying = sub_adv_carrying[sub_adv_carrying["capacity"]==number]
                    standard_capacity = [item*vert_yeild for item in list(subsub_adv_carrying["capacity"])]
                    avelable_initial_material = list(subsub_adv_carrying["factory_province_code"].map(sub_surplus_shortage.set_index("کد کارخانه")["مازاد"]).fillna(0))
                    adv_carrying_values = calculate_adv_carrying_values(distance_matrix, standard_capacity, avelable_initial_material, transferring_cost)
                    subsub_adv_carrying["year"] = year
                    subsub_adv_carrying["carrying_number"] = [item /vert_yeild_feed for item in adv_carrying_values]
                    subsub_adv_carrying_total = pd.concat([subsub_adv_carrying_total, subsub_adv_carrying], axis=0)                 
    subsub_adv_carrying_total.insert(1, "city", subsub_adv_carrying_total["factory_province_code"].\
                                     map(plants_dis.set_index("factory_province_code")["province"]))
    subsub_adv_carrying_total.rename(columns={"factory_province_code": "plant_code"}, inplace=True)
    subsub_adv_carrying_total_drop_duplicates= subsub_adv_carrying_total.drop_duplicates(subset=["city", "product_code", "capacity",\
                                                      	"status","carrying_number"])
    subsub_adv_carrying_total.to_csv("output/adv_carrying.csv", encoding='utf-8-sig', index=False)
    subsub_adv_carrying_total_drop_duplicates.to_csv("output/adv_carrying_drop_duplicates.csv", encoding='utf-8-sig', index=False)

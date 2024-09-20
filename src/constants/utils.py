import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ISCO_CODES = [
    "0110",
    "0210",
    "0310",
    "1111",
    "1112",
    "1113",
    "1114",
    "1120",
    "1211",
    "1212",
    "1213",
    "1219",
    "1221",
    "1222",
    "1223",
    "1311",
    "1312",
    "1321",
    "1322",
    "1323",
    "1324",
    "1330",
    "1341",
    "1342",
    "1343",
    "1344",
    "1345",
    "1346",
    "1349",
    "1411",
    "1412",
    "1420",
    "1431",
    "1439",
    "2111",
    "2112",
    "2113",
    "2114",
    "2120",
    "2131",
    "2132",
    "2133",
    "2141",
    "2142",
    "2143",
    "2144",
    "2145",
    "2146",
    "2149",
    "2151",
    "2152",
    "2153",
    "2161",
    "2162",
    "2163",
    "2164",
    "2165",
    "2166",
    "2211",
    "2212",
    "2221",
    "2222",
    "2230",
    "2240",
    "2250",
    "2261",
    "2262",
    "2263",
    "2264",
    "2265",
    "2266",
    "2267",
    "2269",
    "2310",
    "2320",
    "2330",
    "2341",
    "2342",
    "2351",
    "2352",
    "2353",
    "2354",
    "2355",
    "2356",
    "2359",
    "2411",
    "2412",
    "2413",
    "2421",
    "2422",
    "2423",
    "2424",
    "2431",
    "2432",
    "2433",
    "2434",
    "2511",
    "2512",
    "2513",
    "2514",
    "2519",
    "2521",
    "2522",
    "2523",
    "2529",
    "2611",
    "2612",
    "2619",
    "2621",
    "2622",
    "2631",
    "2632",
    "2633",
    "2634",
    "2635",
    "2636",
    "2641",
    "2642",
    "2643",
    "2651",
    "2652",
    "2653",
    "2654",
    "2655",
    "2656",
    "2659",
    "3111",
    "3112",
    "3113",
    "3114",
    "3115",
    "3116",
    "3117",
    "3118",
    "3119",
    "3121",
    "3122",
    "3123",
    "3131",
    "3132",
    "3133",
    "3134",
    "3135",
    "3139",
    "3141",
    "3142",
    "3143",
    "3151",
    "3152",
    "3153",
    "3154",
    "3155",
    "3211",
    "3212",
    "3213",
    "3214",
    "3221",
    "3222",
    "3230",
    "3240",
    "3251",
    "3252",
    "3253",
    "3254",
    "3255",
    "3256",
    "3257",
    "3258",
    "3259",
    "3311",
    "3312",
    "3313",
    "3314",
    "3315",
    "3321",
    "3322",
    "3323",
    "3324",
    "3331",
    "3332",
    "3333",
    "3334",
    "3339",
    "3341",
    "3342",
    "3343",
    "3344",
    "3351",
    "3352",
    "3353",
    "3354",
    "3355",
    "3359",
    "3411",
    "3412",
    "3413",
    "3421",
    "3422",
    "3423",
    "3431",
    "3432",
    "3433",
    "3434",
    "3435",
    "3511",
    "3512",
    "3513",
    "3514",
    "3521",
    "3522",
    "4110",
    "4120",
    "4131",
    "4132",
    "4211",
    "4212",
    "4213",
    "4214",
    "4221",
    "4222",
    "4223",
    "4224",
    "4225",
    "4226",
    "4227",
    "4229",
    "4311",
    "4312",
    "4313",
    "4321",
    "4322",
    "4323",
    "4411",
    "4412",
    "4413",
    "4414",
    "4415",
    "4416",
    "4419",
    "5111",
    "5112",
    "5113",
    "5120",
    "5131",
    "5132",
    "5141",
    "5142",
    "5151",
    "5152",
    "5153",
    "5161",
    "5162",
    "5163",
    "5164",
    "5165",
    "5169",
    "5211",
    "5212",
    "5221",
    "5222",
    "5223",
    "5230",
    "5241",
    "5242",
    "5243",
    "5244",
    "5245",
    "5246",
    "5249",
    "5311",
    "5312",
    "5321",
    "5322",
    "5329",
    "5411",
    "5412",
    "5413",
    "5414",
    "5419",
    "6111",
    "6112",
    "6113",
    "6114",
    "6121",
    "6122",
    "6123",
    "6129",
    "6130",
    "6210",
    "6221",
    "6222",
    "6223",
    "6224",
    "6310",
    "6320",
    "6330",
    "6340",
    "7111",
    "7112",
    "7113",
    "7114",
    "7115",
    "7119",
    "7121",
    "7122",
    "7123",
    "7124",
    "7125",
    "7126",
    "7127",
    "7131",
    "7132",
    "7133",
    "7211",
    "7212",
    "7213",
    "7214",
    "7215",
    "7221",
    "7222",
    "7223",
    "7224",
    "7231",
    "7232",
    "7233",
    "7234",
    "7311",
    "7312",
    "7313",
    "7314",
    "7315",
    "7316",
    "7317",
    "7318",
    "7319",
    "7321",
    "7322",
    "7323",
    "7411",
    "7412",
    "7413",
    "7421",
    "7422",
    "7511",
    "7512",
    "7513",
    "7514",
    "7515",
    "7516",
    "7521",
    "7522",
    "7523",
    "7531",
    "7532",
    "7533",
    "7534",
    "7535",
    "7536",
    "7541",
    "7542",
    "7543",
    "7544",
    "7549",
    "8111",
    "8112",
    "8113",
    "8114",
    "8121",
    "8122",
    "8131",
    "8132",
    "8141",
    "8142",
    "8143",
    "8151",
    "8152",
    "8153",
    "8154",
    "8155",
    "8156",
    "8157",
    "8159",
    "8160",
    "8171",
    "8172",
    "8181",
    "8182",
    "8183",
    "8189",
    "8211",
    "8212",
    "8219",
    "8311",
    "8312",
    "8321",
    "8322",
    "8331",
    "8332",
    "8341",
    "8342",
    "8343",
    "8344",
    "8350",
    "9111",
    "9112",
    "9121",
    "9122",
    "9123",
    "9129",
    "9211",
    "9212",
    "9213",
    "9214",
    "9215",
    "9216",
    "9311",
    "9312",
    "9313",
    "9321",
    "9329",
    "9331",
    "9332",
    "9333",
    "9334",
    "9411",
    "9412",
    "9510",
    "9520",
    "9611",
    "9612",
    "9613",
    "9621",
    "9622",
    "9623",
    "9624",
    "9629",
]

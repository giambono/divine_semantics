# divine_semantics

Unlocking Dante's Divine Comedy with NLP-Powered Semantic Search

# optimized weights
models = {"multilingual_e5": SentenceTransformer("intfloat/multilingual-e5-large")}
df = df[(df["cantica_id"] == 1) & (df["type_id"] == 1)]

{'dante': 0.3330178901068889, 
'musa': 0.07467696196887205, 
'kirkpatrick': 0.3430391689652815, 
'durling': 0.24926597895895738}

# pages to extract:

# prompt

extract only the english translation of the canto from the pdf attached and print it 3 verses per line
no numeration. and print it all. each row you print must contain three verses
print it so that i can paste it each tercet in one cell in excel

# recalled verses, simulation

i will be your boss
I will be your team leader
I’ll be the one guiding you
I’ll be your point of contact
I am your point of contact
I propose myself like your point of contact

in the lake of the heart
Within the heart's waters

my soul running away
it was the starting of the morning
it was the beginning of the day

why imperato r is bulling

o your among us the best

# Divine Comedy Database

## 1. Setup the Database

Run the following commands:

```sh
psql -U postgres -d divine_comedy -f database/schema.sql
psql -U postgres -d divine_comedy -f database/seed_data.sql
psql -U postgres -d divine_comedy -f database/insert_data.sql

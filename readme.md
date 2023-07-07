## MatIE

This is a simple Information Extraction (IE) system for Material Science text analysis (Entity Extraction + Relation Extraction).

### Setup

Clone this repo:

    git clone https://github.com/zzsfornlp/MatIE src

Prepare the environment using conda:

    conda create -n matie python=3.8
    conda install numpy scipy cython pybind11 pandas pip nltk
    pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    pip install transformers==4.8.2 datasets stanza scikit-learn
    pip install pymongo

Before running anything, make sure to export the src directory to your `$PYTHONPATH`:

    export PYTHONPATH=/your/path/to/src

### Data Preparation

We utilize our own json (JSON Lines) data format, and we also provide a script to convert BRAT files to our format. To convert all the BRAT `*.ann` files from an `${INPUT_FOLDER}` to an output file `${OUTPUT_FILE}`, utilize:

    python3 -m mspx.tools.al.utils_brat cmd:b2z input_path:${INPUT_FOLDER}/ output_path:${OUTPUT_FILE} delete_nils:1

The other direction is also available:

    python3 -m mspx.tools.al.utils_brat cmd:z2b input_path:${INPUT_FILE} output_path:${OUTPUT_DIR}/ delete_nils:1

### Obtain MatBERT

Our system is based on MatBERT, please download it first (https://github.com/lbnlp/MatBERT):

    mkdir -p matbert-base-cased
    MODEL_PATH=matbert-base-cased
    curl -# -o $MODEL_PATH/config.json https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_cased_30522_wd/config.json
    curl -# -o $MODEL_PATH/vocab.txt https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_cased_30522_wd/vocab.txt
    curl -# -o $MODEL_PATH/pytorch_model.bin https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_cased_30522_wd/pytorch_model.bin

### Build Vocab

Before running, we first need to build vocabs (using data file `${DATA_FILE}`):

    mkdir -p vpack_mat
    cd vpack_mat
    python3 -m mspx.tasks.zrel.main log_file:_log fs:build train0.group_files:${DATA_FILE} fs:build conf_sbase:bert_name:__matbert-base-cased::cateH:ef::cateT:ef::modH:ef

### Training and Testing

Finally, we can train and test an IE (entity+relation) model, assuming we have train/dev/test files of `${TRAIN_DATA}`, `${DEV_DATA}` and `${TEST_DATA}`:

    mkdir -p run
    cd run
    CARGS="conf_output:_conf log_stderr:1 log_file:_log device:0 fs:build,train,test vocab_load_dir:__vpack_mat/ rel0.eval.match_arg_with_frame_type:0 conf_sbase:bert_name:__matbert-base-cased::cateH:ef::cateT:ef::modH:ef extH.train_cons:1 extH.pred_cons:1 conf_sbase2:base_layer:8 mark_extra.init_scale:0.25 rel0.neg_strg_thr:0.99 rel0.neg_rate.val:10 rel0.neg_rate.ff:0.1*i+0.1 pred_dec_cons:mat train0.inst_f:sentF train1.inst_f:sentF"
    CUDA_VISIBLE_DEVICES=0 python3 -m mspx.tasks.zrel.main ${CARGS} train0.group_files:${TRAIN_DATA} dev0.group_files:${DEV_DATA} test0.group_files:${TEST_DATA} test1.group_files:${TEST_DATA} test1.tasks:enc0,rel0

We can also perform predictions with a trained model (assuming training dir is `${MODEL_DIR}`) (from `${INPUT_DATA}` to `${OUTPUT_DATA}`):

    python3 -m mspx.tasks.zrel.main ${MODEL_DIR}/_conf model_load_name:${MODEL_DIR}/zmodel.best.m vocab_load_dir:__vpack_mat/ log_stderr:1 fs:test d_input_dir: test1.group_files: test0.test_streaming:5 test0.group_files:${INPUT_DATA} test0.output_file:${OUTPUT_DATA}

### Easier Decoding

We also provide a script `decode.sh` for easier decoding that takes a directory of "*.txt" files as input and produces BRAT's "*.ann" annotations to an output directory. To use this script, we need to provide a model dir `${MODEL_DIR}` and a vocab dir `${VOCAB_DIR}`. Also prepare a `matbert-base-cased` dir (that contains the matbert files) at the current working directory. See previous sections on how to prepare them. With all of these prepared, we can run the decoding script (input from `${INPUT_DIR}` and output to `${OUTPUT_DIR}`):

    CUDA_VISIBLE_DEVICES=<YourGpuID> MODEL_DIR=$MODEL_DIR VOCAB_DIR=$VOCAB_DIR INPUT_DIR=$INPUT_DIR OUTPUT_DIR=$OUTPUT_DIR EXTRA_ARGS= bash decode.sh

### Import Data into MongoDB for easier querying

Setting up with the following steps:

- Install MongoDB, please check: https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/
- Configure the conf file `mongod.conf`: replace the `storage.dbPath` and `systemLog.dbPath` with your real paths.
- Setup auth: https://www.mongodb.com/docs/manual/tutorial/configure-scram-client-authentication/
- Start a server locally (possible with a plain user): `mongod -f mongod.conf --auth --fork`.
- Setup db (assuming we are using a db named `mat`):

(Use the following to setup):

    mongosh -u "myUserAdmin" -p
    use mat
    db.createUser(
      {
        user: "mat",
        pwd:  passwordPrompt(),   // or cleartext password
        roles: [ { role: "readWrite", db: "mat" } ]
      }
    )

Use the following command to import/export documents:
(For a data collection `data`, we have three collections for it: 1) `data`: original document information, 2) `data_ent`: entity information, 3) `data_rel`: relation information.)

    # To import documents from input data-file "data.json" to DB Collections "mat.data*"
    python3 -m mspx.tools.al.utils_mdb "uri:mongodb://mat:${PASSWORD}@localhost:27017/mat" "cmd:update data data.json"
    
    # To explore documents from Collections "mat.data*" to output data-file "data2.json"
    python3 -m mspx.tools.al.utils_mdb "uri:mongodb://mat:${PASSWORD}@localhost:27017/mat" "cmd:read data data2.json"
    
    # We can also perform various queries with MongoDB, first connect with mongosh:
    mongosh -u "mat" --authenticationDatabase "mat" -p
    use mat

Here are some example queries:

    # Example query: list all the entity/relation types (sorted by frequency)
    db.data_ent.aggregate([
        {$group: {_id: "$label", count: {$count: {}}}}, 
        {$sort: {count: -1}}
    ])
    db.data_rel.aggregate([
        {$group: {_id: "$label", count: {$count: {}}}}, 
        {$sort: {count: -1}}
    ])

    # Example query: list all the "Descriptor" entity's mention texts (appearance >1 and sorted by frequency)
    db.data_ent.aggregate([
        {$match: {label: "Descriptor"}}, 
        {$group: {_id: "$text", count: {$count: {}}}}, 
        {$sort: {count: -1}}, 
        {$match: {count: {$gte: 2}}}
    ])

    # Example query: list all the entity mentions containing the string "mech" (grouped by label & text)
    db.data_ent.aggregate([
        {$match: {text: {$regex: /.*mech.*/}}}, 
        {$group: {_id: ["$label", "$text"], count: {$count: {}}}}, 
        {$sort: {count: -1}}
    ])

    # Example query: list all the relation types that a "Descriptor" entity is involved
    db.data_rel.aggregate([
        {$lookup: {
            from: "data_ent",
            let: {did: "$doc_id", eid: "$arg0"},
            pipeline: [{ $match: { $expr: { $and: [
                { $eq: [ "$doc_id",  "$$did" ] }, { $eq: [ "$frame_id", "$$eid" ] }
            ]}}}],
            as: "ent0"
            }
        },
        {$lookup: {
            from: "data_ent",
            let: {did: "$doc_id", eid: "$arg1"},
            pipeline: [{ $match: { $expr: { $and: [
                { $eq: [ "$doc_id",  "$$did" ] }, { $eq: [ "$frame_id", "$$eid" ] }
            ]}}}],
            as: "ent1"
            }
        },
        {$match: {$or: [{'ent0.0.label': "Descriptor"}, {'ent1.0.label': "Descriptor"}] }},
        {$group: {_id: ["$label"], count: {$count: {}}}}, 
        {$sort: {count: -1}}
    ] )

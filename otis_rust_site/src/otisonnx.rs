// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
// Copyright 2019 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use rust_bert::bert::{BertConfigResources, BertModelResources, BertVocabResources};
use rust_bert::pipelines::common::{ModelResource, ModelType};
use rust_bert::pipelines::ner::NERModel;
use rust_bert::pipelines::sentiment::{SentimentModel, SentimentConfig};
use rust_bert::pipelines::token_classification::{
    LabelAggregationOption, TokenClassificationConfig,
};
use rust_bert::pipelines::zero_shot_classification::ZeroShotClassificationModel;
use rust_bert::resources::{RemoteResource, LocalResource};


pub fn test() -> anyhow::Result<()> {
    let sentiment_classifier = SentimentModel::new( SentimentConfig { model_type: todo!(), model_resource: todo!(), config_resource: todo!(), vocab_resource: todo!(), merges_resource: todo!(), lower_case: todo!(), strip_accents: todo!(), add_prefix_space: todo!(), device: todo!() })?;
                                                        
    let input = [
        "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring.",
        "This film tried to be too many things all at once: stinging political satire, Hollywood blockbuster, sappy romantic comedy, family values promo...",
        "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.",
    ];

    let output = sentiment_classifier.predict(&input);

    Ok(())
}
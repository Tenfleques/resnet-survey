import React, { Component } from 'react';
import Select from "./Select"
import Image from "./Image"
import HyperParams from "./hyper_params"
import pkg from "../Configs/package"


class Stage extends Component {
    constructor(props) {
        super(props);
        this.state = {
            default_hp : JSON.parse(JSON.stringify(HyperParams)),
            hyper_params : JSON.parse(JSON.stringify(HyperParams)),
            images : {
                root : "charts/",
                structures: "structures/"
            }
        }
        this.resetVariables = this.resetVariables.bind(this)
        this.handleSelectChange = this.handleSelectChange.bind(this)
    }

    renderImageControl(phase="train"){
        let collection = this.collectImageNames(this.state.images, this.state.hyper_params);
        return collection[phase].length ? collection[phase] : collection["base_" + phase]
    }
    resetVariables(){
        console.log(this.state.default_hp);
        this.setState({hyper_params : JSON.parse(JSON.stringify(this.state.default_hp))})
    }
    collectImageNames(images, hyper_params){
        let root =  images.root 
                    + "resnet-" + hyper_params.layers.value + "/"
                    + "cifar-" + hyper_params.cifar.value 
                    ;
        let test_images = [];
        let train_images = [];
        let metrics = [
                "_loss_val_mean",
                "_acc_val_mean"
            ];
       
        let base_train = [], base_test = [];

        for (let i = 0; i < metrics.length; ++i){
            let across_structures_src_train = images.structures 
                                + "cifar-" + hyper_params["cifar"].value + "/"
                                + "train-20_50_56_62_110-"
                                + metrics[i].substring(1)
            let across_structures_src_test = images.structures 
                                + "cifar-" + hyper_params["cifar"].value + "/"
                                + "test-20_50_56_62_110-"
                                + metrics[i].substring(1)


            base_train.push(<Image key={"structure-cross-image" + metrics[i]} src={across_structures_src_train + ".png"} name={"Train reference"}/>);

            base_test.push(<Image key={"structure-cross-image" + metrics[i]} src={across_structures_src_test + ".png"} name={"Test reference"}/>);

            for (let hp in hyper_params){
                if(
                    (hyper_params[hp].is_image ||  hyper_params[hp].value == 50 || hyper_params[hp].value == 62) 
                    && hyper_params[hp].value){
                
                    train_images.push(<Image key={"structure-cross-image-hyper" + metrics[i] + hp} src={across_structures_src_test + hyper_params[hp].value + ".png"} name={"@" + hp + " " + hyper_params[hp].value }/>);

                    test_images.push(<Image key={"structure-cross-image-hyper" + metrics[i] + hp} src={across_structures_src_test + hyper_params[hp].value + ".png"} name={"@ " + hp + " " + hyper_params[hp].value }/>); 

                    
                    
                    let name = hyper_params[hp].value + metrics[i];
                    if(hp == "layers"){
                        if(hyper_params[hp].value == 50){
                            name = "-l50" + metrics[i];
                        }

                        if (hyper_params[hp].value == 62){
                            name = "-l62" + metrics[i];
                        }
                    }
                    
                    

                    let src = root + "-test_" + name + ".png"
    
                    let v_name = name.split("_")[0]

                    let test_image = <Image key={"test-image-" + i + hp} src={src} name={"Test " + v_name}/>
                    test_images.push(test_image);


                    let src_train = root + "-train_" + name + ".png"

                    v_name = name.split("_")[0]
    
                    let train_image = <Image key ={"train-image-" + i + hp} src={src_train} name={"Train " + v_name}/>
                    train_images.push(train_image);
                } 
            }    
        }    
        return {
            train :train_images,
            test : test_images,
            base_train,
            base_test
        }
    }
    renderHyperControls(){
        let selects = []
        for (var i in this.state.hyper_params){

            selects.push(<Select 
                            key={"select-" + i} 
                            select={this.state.hyper_params[i]} 
                            onChange={this.handleSelectChange}
                            name={i}
                            />)
        }
        return selects;
    }
    handleSelectChange(ev){
        let hyper_params = this.state.hyper_params;

        hyper_params[ev.target.name].value = ev.target.options[ev.target.selectedIndex].value;

        this.setState({hyper_params})
    }
    render() {
    
        return (
            <div className="row">
                <div className="bg-primary col-12 pt-3">
                    <h3 className="col-12 text-white">
                        Hyper-parameter investigation for {pkg.appname}
                    </h3>
                    <p className="text-lead col-12 text-white">
                        {pkg.description}
                    </p>
                </div>
                
                <div className="knob-box col-12 col-md-6 col-lg-3 ">
                    <button onClick={() => this.resetVariables()} className="btn btn-transparent border-primary col-12  my-3">
                        reset to reference
                    </button>
                    <div className="border-top py-3">
                        {this.renderHyperControls()}
                    </div>
                    
                </div>
                <div className="col-12 col-md-6 col-lg-9 
                 mh-85 mh-md-85 y-scroll-auto">
                    <div className="row">
                        <div className={"" + "invisible pr-5 col-12 text-right"}>
                            <button onClick={() => this.stepBack()} className="btn btn-transparent border-primary my-3 mr-4">
                                <span className="h5">
                                    &lt;
                                </span>
                            </button>
                        </div>
                        <div className="col-md-10 col-lg-6 pr-0">
                            {this.renderImageControl("train")}
                        </div> 
                        <div className="col-md-10 col-lg-6 pr-0">
                            {this.renderImageControl("test")}
                        </div>                        
                    </div>
                </div>                
            </div>
        )
    }
}

export default Stage;
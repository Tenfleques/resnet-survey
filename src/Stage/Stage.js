import React, { Component } from 'react';
import Select from "./Select"
import Image from "./Image"
import hyper_params from "./hyper_params"
import pkg from "../Configs/package"


class Stage extends Component {
    constructor(props) {
        super(props);
        this.state = {
            hyper_params,
            images : {
                root : "charts/"
            }
        }
        this.handleSelectChange = this.handleSelectChange.bind(this)
    }

    renderImageControl(phase="train"){
        return this.collectImageNames(this.state.images, this.state.hyper_params)[phase];
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
        
        for (let hp in hyper_params){
            if(hyper_params[hp].is_image && hyper_params[hp].value){
                for (let i = 0; i < metrics.length; ++i){
                    let name = hyper_params[hp].value + metrics[i];
                    if(hp == "layers"){
                        if(hyper_params[hp].value == 50){
                            name = "-l50" + metrics[i];
                        }

                        if (hyper_params[hp].value == 62){
                            name = "-l62" + metrics[i];
                        }
                    }
                    
                    

                    let src = root + "-test_" +name + ".png"
    
                    let v_name = name.split("_")[0]

                    let test_image = <Image key={"test-image-" + i + hp} src={src} name={"Test " + v_name}/>
                    test_images.push(test_image);


                    let src_train = root + "-train_" + name + ".png"

                    v_name = name.split("_")[0]
    
                    let train_image = <Image key ={"test-image-" + i + hp} src={src_train} name={"Train " + v_name}/>
                    train_images.push(train_image);
                } 
            }    
        }    
        return {
            train :train_images,
            test : test_images
        }
    }
    renderHyperControls(){
        let selects = []
        for (var i in this.state.hyper_params){

            selects.push(<Select 
                            key={"select-" + i} 
                            select={hyper_params[i]} 
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
                
                <div className="knob-box col-12 col-md-6 col-lg-3 mt-4">
                    {this.renderHyperControls()}
                </div>
                <div className="col-12 col-md-6 col-lg-9 mh-85 mh-md-85 y-scroll-auto">
                    <div className="row">
                        <div className="col-md-10 col-lg-6">
                            {this.renderImageControl("train")}
                        </div> 
                        <div className="col-md-10 col-lg-6">
                            {this.renderImageControl("test")}
                        </div>                        
                    </div>
                </div>                
            </div>
        )
    }
}

export default Stage;
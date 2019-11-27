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
                root : "/logs/charts/"
            }
        }
        this.handleSelectChange = this.handleSelectChange.bind(this)
    }

    renderImageControl(){
        return this.collectImageNames(this.state.images, this.state.hyper_params);
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

                    let src = root + "-test_" +name + ".png"
    
                    let test_image = <Image key={"test-image-" + i + hp} src={src} name={"Test " + name}/>
                    test_images.push(test_image);


                    let src_train = root + "-train_" + name + ".png"
    
                    let train_image = <Image key ={"test-image-" + i + hp} src={src_train} name={"Train " + name}/>
                    train_images.push(train_image);
                } 
            }    
        }    
        return (
            train_images,
            test_images
        )
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
                <div className="col-12 col-md-6 col-lg-9">
                    <div className="row">
                        {this.renderImageControl()}
                    </div>
                </div>                
            </div>
        )
    }
}

export default Stage;
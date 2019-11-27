(this["webpackJsonpview-resnet-charts"]=this["webpackJsonpview-resnet-charts"]||[]).push([[0],[,,,,,,,function(e){e.exports=JSON.parse('{"cifar":{"value":"10","caption":"cifar","is_image":false,"options":[{"label":"10","value":"10"},{"label":"100","value":"100"}]},"layers":{"value":"20","caption":"layers","is_image":false,"options":[{"label":"20","value":"20"},{"label":"50","value":"50"},{"label":"56","value":"56"},{"label":"62","value":"62"},{"label":"110","value":"110"}]},"lr":{"value":"-lr-.01","caption":"learning rate","is_image":true,"options":[{"label":"0.01","value":"-lr-.01"},{"label":"0.02","value":"-lr-.02"},{"label":"0.2","value":"-lr-.2"},{"label":"0.5","value":"-lr-.5"},{"label":"1.0","value":"-lr-1.0"}]},"epochs":{"value":"-epochs-160","caption":"number of epochs","is_image":true,"options":[{"label":"80","value":"-epochs-80"},{"label":"160","value":"-epochs-160"},{"label":"320","value":"-epochs-320"}]},"mb":{"value":"-mini-batch-128","caption":"mini-batch size","is_image":true,"options":[{"label":"64","value":"-mini-batch-64"},{"label":"128","value":"-mini-batch-128"},{"label":"256","value":"-mini-batch-256"}]},"rb":{"value":"","caption":"residual blocks","is_image":true,"options":[{"label":"-1","value":""},{"label":"1","value":"-rb-2"},{"label":"2","value":"-rb-2"}]},"sc":{"value":"","caption":"skip connections","is_image":true,"options":[{"label":"-1","value":""},{"label":"1","value":"-sc-1"},{"label":"3","value":"-sc-3"}]},"tds":{"value":"","caption":"training dataset","is_image":true,"options":[{"label":"1","value":""},{"label":"0.5","value":"-tds-0.5"},{"label":"0.2","value":"-tds-0.2"}]}}')},function(e){e.exports=JSON.parse('{"appname":"ResNet survey","description":"Investigating perfomances of ResNet implementation on CIFAR-10 and CIFAR-100 by tuning hyper-parameters"}')},,,function(e,a,t){e.exports=t(19)},,,,,function(e,a,t){},function(e,a,t){},function(e,a,t){},function(e,a,t){"use strict";t.r(a);var l=t(0),n=t.n(l),r=t(10),s=t.n(r),i=(t(16),t(1)),o=t(2),c=t(6),u=t(4),m=t(5),v=t(3),p=function(e){return n.a.createElement("div",{className:"form-group"},n.a.createElement("label",{htmlFor:e.name},e.select.caption),n.a.createElement("select",{className:"form-control",onChange:e.onChange,id:e.name,name:e.name,defaultValue:e.select.value},e.select.options.map((function(e,a){return n.a.createElement("option",{key:a,value:e.value},e.label)}))))};var h=function(e){return function(e){var a=new XMLHttpRequest;return a.open("HEAD",e,!1),a.send(),404!==a.status}(e.src)?n.a.createElement("div",{className:"col-12"},n.a.createElement("h5",null,e.name),n.a.createElement("img",{className:"img-thumbnail border-0 img-fluid  float-right",src:e.src,alt:""})):""},b=t(7),d=t(8),g=function(e){function a(e){var t;return Object(i.a)(this,a),(t=Object(c.a)(this,Object(u.a)(a).call(this,e))).state={hyper_params:b,images:{root:"charts/"}},t.handleSelectChange=t.handleSelectChange.bind(Object(v.a)(t)),t}return Object(m.a)(a,e),Object(o.a)(a,[{key:"renderImageControl",value:function(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:"train";return this.collectImageNames(this.state.images,this.state.hyper_params)[e]}},{key:"collectImageNames",value:function(e,a){var t=e.root+"resnet-"+a.layers.value+"/cifar-"+a.cifar.value,l=[],r=[],s=["_loss_val_mean","_acc_val_mean"];for(var i in a)if(a[i].is_image&&a[i].value)for(var o=0;o<s.length;++o){var c=a[i].value+s[o],u=t+"-test_"+c+".png",m=c.split("_")[0],v=n.a.createElement(h,{key:"test-image-"+o+i,src:u,name:"Test "+m});l.push(v);var p=t+"-train_"+c+".png";m=c.split("_")[0];var b=n.a.createElement(h,{key:"test-image-"+o+i,src:p,name:"Train "+m});r.push(b)}return{train:r,test:l}}},{key:"renderHyperControls",value:function(){var e=[];for(var a in this.state.hyper_params)e.push(n.a.createElement(p,{key:"select-"+a,select:b[a],onChange:this.handleSelectChange,name:a}));return e}},{key:"handleSelectChange",value:function(e){var a=this.state.hyper_params;a[e.target.name].value=e.target.options[e.target.selectedIndex].value,this.setState({hyper_params:a})}},{key:"render",value:function(){return n.a.createElement("div",{className:"row"},n.a.createElement("div",{className:"bg-primary col-12 pt-3"},n.a.createElement("h3",{className:"col-12 text-white"},"Hyper-parameter investigation for ",d.appname),n.a.createElement("p",{className:"text-lead col-12 text-white"},d.description)),n.a.createElement("div",{className:"knob-box col-12 col-md-6 col-lg-3 mt-4"},this.renderHyperControls()),n.a.createElement("div",{className:"col-12 col-md-6 col-lg-9 mh-85 mh-md-85 y-scroll-auto"},n.a.createElement("div",{className:"row"},n.a.createElement("div",{className:"col-md-10 col-lg-6"},this.renderImageControl("train")),n.a.createElement("div",{className:"col-md-10 col-lg-6"},this.renderImageControl("test")))))}}]),a}(l.Component),f=(t(17),t(18),function(e){function a(){return Object(i.a)(this,a),Object(c.a)(this,Object(u.a)(a).apply(this,arguments))}return Object(m.a)(a,e),Object(o.a)(a,[{key:"render",value:function(){return n.a.createElement("div",{className:"App container-fluid"},n.a.createElement(g,null))}}]),a}(l.Component));Boolean("localhost"===window.location.hostname||"[::1]"===window.location.hostname||window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/));s.a.render(n.a.createElement(f,null),document.getElementById("root")),"serviceWorker"in navigator&&navigator.serviceWorker.ready.then((function(e){e.unregister()}))}],[[11,1,2]]]);
//# sourceMappingURL=main.db6429de.chunk.js.map
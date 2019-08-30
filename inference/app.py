from flask import Flask, render_template, url_for, request, redirect
# from flask_wtf import Form
from run_inf import QBert
app = Flask(__name__)



@app.route("/")
def home():
    
    # if query:
    #     show_list = model.get_answer(q = query)
        

    # else:
    show_list = False

    return render_template("home.html", texts = show_list)


@app.route("/search")
def search():
    query = request.args.get("q")
    
    if query:
        #print(query)
        show_list = model.get_answer(q = query)
        #show_list = model.get_answer()
        
        
    else:
        return redirect(url_for(home))

    return render_template("home.html", texts = show_list, query=query)




# @app.route("/results")
# def results():
#     return render_template("index.html")













if __name__ == "__main__":
    model = QBert()
    app.run(debug=True, use_reloader=False)
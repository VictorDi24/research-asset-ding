

def get_cases(results_folder):
    from objectives_optimize import loop_optimization, portfolio, elastic_net, berhu_penalty,huber_penalty

    cases = {
        # ---------------
        # --------------
        # ELASTIC NET WITH LB =  0.1
        # --------------
        # --------------

        "berhu_a_base": (
            loop_optimization,
            {
                "start_day": "2016-01-02",
                "end_day": "2016-01-21",
                "log_every": 10,
                "portfolio_class": berhu_penalty,
                "portfolio_kwargs": {
                    "scale" : 0.005,
                    "threshold" : 1E-4,
                },
            }
        ),
        "berhu_b_base": (
            loop_optimization,
            {
                "start_day": "2016-01-02",
                "end_day": "2016-01-21",
                "log_every": 10,
                "portfolio_class": berhu_penalty,
                "portfolio_kwargs": {
                    "scale" : 0.01,
                    "threshold" : 1E-4,
                    "total_turnover": 0.3
                },
            }
        ),
        "berhu_c_base": (
            loop_optimization,
            {
                "start_day": "2016-01-02",
                "end_day": "2016-01-21",
                "log_every": 10,
                "portfolio_class": berhu_penalty,
                "portfolio_kwargs": {
                    "scale" : 0.0001,
                    "threshold" : 1E-4,
                },
            }
        ),
        "elastic_a_full_4": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.005,
                    "lambda_b" : 0.1,
                },
            }
        ),
        "elastic_b_full_4": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.01,
                    "lambda_b" : 0.1,
                },
            }
        ),
        "elastic_c_full_4": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.02,
                    "lambda_b" : 0.1,
                },
            }
        ),
        "elastic_d_full_4": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.03,
                    "lambda_b" : 0.1,
                },
            }
        ),
        "elastic_e_full_4": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.04,
                    "lambda_b" : 0.1,
                },
            }
        ),
        "elastic_f_full_4": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.05,
                    "lambda_b" : 0.1,
                },
            }
        ),
        "elastic_g_full_4": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.06,
                    "lambda_b" : 0.1,
                },
            }
        ),

        # ---------------
        # --------------
        # ELASTIC NET WITH LB =  0.5
        # --------------
        # --------------
        "elastic_a_full_5": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.005,
                    "lambda_b" : 0.5,
                },
            }
        ),
        "elastic_b_full_5": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.01,
                    "lambda_b" : 0.5,
                },
            }
        ),
        "elastic_c_full_5": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.02,
                    "lambda_b" : 0.5,
                },
            }
        ),
        "elastic_d_full_5": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.03,
                    "lambda_b" : 0.5,
                },
            }
        ),
        "elastic_e_full_5": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.04,
                    "lambda_b" : 0.5,
                },
            }
        ),
        "elastic_f_full_5": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.05,
                    "lambda_b" : 0.5,
                },
            }
        ),
        "elastic_g_full_5": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.06,
                    "lambda_b" : 0.5,
                },
            }
        ),
        # ---------------
        # --------------
        # ELASTIC NET WITH LB =  1
        # --------------
        # --------------
        "elastic_a_full_6": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.005,
                    "lambda_b" : 1,
                },
            }
        ),
        "elastic_b_full_6": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.01,
                    "lambda_b" : 1,
                },
            }
        ),
        "elastic_c_full_6": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.02,
                    "lambda_b" : 1,
                },
            }
        ),
        "elastic_d_full_6": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.03,
                    "lambda_b" : 1,
                },
            }
        ),
        "elastic_e_full_6": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.04,
                    "lambda_b" : 1,
                },
            }
        ),
        "elastic_f_full_6": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.05,
                    "lambda_b" : 1,
                },
            }
        ),
        "elastic_g_full_6": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.06,
                    "lambda_b" : 1,
                },
            }
        ),
        # ---------------
        # --------------
        # ELASTIC NET WITH LB =  0.7
        # --------------
        # --------------
        "elastic_a_full_7": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.005,
                    "lambda_b" : 0.7,
                },
            }
        ),
        "elastic_b_full_7": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.01,
                    "lambda_b" : 0.7,
                },
            }
        ),
        "elastic_c_full_7": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.02,
                    "lambda_b" : 0.7,
                },
            }
        ),
        "elastic_d_full_7": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.03,
                    "lambda_b" : 0.7,
                },
            }
        ),
        "elastic_e_full_7": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.04,
                    "lambda_b" : 0.7,
                },
            }
        ),
        "elastic_f_full_7": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.05,
                    "lambda_b" : 0.7,
                },
            }
        ),
        "elastic_g_full_7": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.06,
                    "lambda_b" : 0.7,
                },
            }
        ),
        # ---------------
        # --------------
        # ELASTIC NET WITH LB =  0.1, INTERSPARSED LA VALUES
        #
        # --------------
        # --------------
        "elastic_a_full_8": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.015,
                    "lambda_b" : 0.1,
                },
            }
        ),
        "elastic_b_full_8": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.025,
                    "lambda_b" : 0.1,
                },
            }
        ),
        "elastic_c_full_8": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.035,
                    "lambda_b" : 0.1,
                },
            }
        ),
        "elastic_d_full_8": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.045,
                    "lambda_b" : 0.1,
                },
            }
        ),
        "elastic_e_full_8": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.055,
                    "lambda_b" : 0.1,
                },
            }
        ),
        # ---------------
        # --------------
        # ELASTIC NET WITH LB =  0.5, INTERSPARSED LA VALUES
        #
        # --------------
        # --------------
        "elastic_a_full_9": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.015,
                    "lambda_b" : 0.5,
                },
            }
        ),
        "elastic_b_full_9": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.025,
                    "lambda_b" : 0.5,
                },
            }
        ),
        "elastic_c_full_9": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.035,
                    "lambda_b" : 0.5,
                },
            }
        ),
        "elastic_d_full_9": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.045,
                    "lambda_b" : 0.5,
                },
            }
        ),
        "elastic_e_full_9": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.055,
                    "lambda_b" : 0.5,
                },
            }
        ),
        # ---------------
        # --------------
        # ELASTIC NET WITH LB =  1, INTERSPARSED LA VALUES
        #
        # --------------
        # --------------
        "elastic_a_full_10": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.015,
                    "lambda_b" : 1,
                },
            }
        ),
        "elastic_b_full_10": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.025,
                    "lambda_b" : 1,
                },
            }
        ),
        "elastic_c_full_10": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.035,
                    "lambda_b" : 1,
                },
            }
        ),
        "elastic_d_full_10": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.045,
                    "lambda_b" : 1,
                },
            }
        ),
        "elastic_e_full_10": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.055,
                    "lambda_b" : 1,
                },
            }
        ),
        "case_a_base": (
            loop_optimization,
            {
                "start_day": "2015-07-01",
                "end_day": "2015-07-21",
                "log_every": 10,
                "portfolio_class": portfolio,
                "portfolio_kwargs": {
                    "alpha_on": False
                },
                "save_filename": results_folder / "results_case_a_base.npz"
            }
        ),
        "case_a_full": (
            loop_optimization,
            {
                "log_every": 100,
                "portfolio_class": portfolio,
                "portfolio_kwargs": {
                    "max_weight": 0.04,
                    "individual_turnover": 0.02,
                    "total_turnover": 0.6,
                    "alpha_on": False
                },
                "save_filename": results_folder / "results_case_a_full.npz"

            }
        ),
        "alpha_a_base": (
            loop_optimization,
            {
                "start_day": "2026-07-01",
                "log_every": 10,
                "portfolio_class": portfolio,
                "portfolio_kwargs": {
                    "max_weight": 0.04,
                    "individual_turnover": 0.02,
                    "total_turnover": 0.4,
                    "alpha" : 0.1,
                    "alpha_on": True,
                    "hard_turnover": 0.2,
                },
                "save_filename": results_folder / "results_alpha_a_base.npz"

            }
        ),
        "alpha_b_base": (
            loop_optimization,
            {
                "start_day": "2026-07-01",
                "log_every": 10,
                "portfolio_class": portfolio,
                "portfolio_kwargs": {
                    "max_weight": 0.04,
                    "individual_turnover": 0.02,
                    "total_turnover": 0.4,
                    "alpha" : 0.00001,
                    "alpha_on": True,
                    "hard_turnover": 0.2,
                },
                "save_filename": results_folder / "results_alpha_b_base.npz"

            }
        ),
        "alpha_c_base": (
            loop_optimization,
            {
                "start_day": "2015-07-01",
                "end_day": "2015-07-21",
                "log_every": 10,
                "portfolio_class": portfolio,
                "portfolio_kwargs": {
                    "max_weight": 0.04,
                    "individual_turnover": 0.02,
                    "total_turnover": 0.4,
                    "alpha" : 0.01,
                    "alpha_on": True,
                    "hard_turnover": 0.3,
                },
                "save_filename": results_folder / "results_alpha_c_base.npz"

            }
        ),
        "alpha_a_full": (
            loop_optimization,
            {
                "log_every": 100,
                "portfolio_class": portfolio,
                "portfolio_kwargs": {
                    "max_weight": 0.04,
                    "individual_turnover": 0.02,
                    "total_turnover": 0.6,
                    "alpha" : 0.1,
                    "alpha_on": True,
                },
                "save_filename": results_folder / "results_alpha_a_full.npz"

            }
        ),
        "alpha_b_full": (
            loop_optimization,
            {
                "start_day" : "2015-07-01",
                "end_day" : "2026-07-01",
                "log_every": 100,
                "portfolio_class": portfolio,
                "portfolio_kwargs": {
                    "max_weight": 0.04,
                    "individual_turnover": 0.02,
                    "total_turnover": 0.6,
                    "alpha" : 0.1,
                    "alpha_on": True,
                },
                "save_filename": results_folder / "results_alpha_b_full.npz"

            }
        ),
        #Checking infeasible day 1926, speed 1
        "check_infeasible": (
            loop_optimization,
            {
                "start_day" : "2022-09-11",
                "end_day" : "2022-09-14",
                "log_every": 10,
                "portfolio_class": portfolio,
                "portfolio_kwargs": {
                    "max_weight": 0.04,
                    "individual_turnover": 0.02,
                    "total_turnover": 0.6,
                    "alpha" : 0.001,
                    "alpha_on": True,
                },
                "save_filename": None

            }
        ),
        "alpha_c_full": (
            loop_optimization,
            {
                "start_day" : "2015-07-01",
                "end_day" : "2026-07-01",
                "log_every": 100,
                "portfolio_class": portfolio,
                "portfolio_kwargs": {
                    "max_weight": 0.04,
                    "individual_turnover": 0.02,
                    "total_turnover": 0.50,
                    "alpha" : 10,
                    "alpha_on": True,
                },
                "save_filename": results_folder / "results_alpha_c_full.npz"

            }
        ),
        "alpha_d_full": (
            loop_optimization,
            {
                "start_day" : "2015-07-01",
                "end_day" : "2026-07-01",
                "log_every": 100,
                "portfolio_class": portfolio,
                "portfolio_kwargs": {
                    "max_weight": 0.04,
                    "individual_turnover": 0.02,
                    "total_turnover": 0.50,
                    "alpha" : 1,
                    "alpha_on": True,
                },
                "save_filename": results_folder / "results_alpha_d_full.npz"

            }
        ),
        "alpha_e_full_hard": (
            loop_optimization,
            {
                "start_day" : "2015-07-01",
                "end_day" : "2026-07-01",
                "log_every": 100,
                "portfolio_class": portfolio,
                "portfolio_kwargs": {
                    "max_weight": 0.04,
                    "individual_turnover": 0.02,
                    "total_turnover": 0.4,
                    "alpha" : 0.001,
                    "alpha_on": True,
                    "hard_turnover": 0.3,
                },
                "save_filename": results_folder / "results_alpha_e_full_hard.npz"

            }
        ),
        "alpha_f_full_hard": (
            loop_optimization,
            {
                "start_day" : "2015-07-01",
                "end_day" : "2026-07-01",
                "log_every": 100,
                "portfolio_class": portfolio,
                "portfolio_kwargs": {
                    "max_weight": 0.04,
                    "individual_turnover": 0.02,
                    "total_turnover": 0.4,
                    "alpha" : 0.01,
                    "alpha_on": True,
                    "hard_turnover": 0.3,
                },
                "save_filename": results_folder / "results_alpha_f_full_hard.npz"

            }
        ),
        "alpha_g_full_hard": (
            loop_optimization,
            {
                "start_day" : "2015-07-01",
                "end_day" : "2026-07-01",
                "log_every": 100,
                "portfolio_class": portfolio,
                "portfolio_kwargs": {
                    "max_weight": 0.04,
                    "individual_turnover": 0.02,
                    "total_turnover": 0.4,
                    "alpha" : 0.1,
                    "alpha_on": True,
                    "hard_turnover": 0.3,
                },
                "save_filename": results_folder / "results_alpha_g_full_hard.npz"

            }
        ),
        "alpha_h_base_hard": (
            loop_optimization,
            {
                "start_day" : "2015-07-01",
                "end_day" : "2015-07-21",
                "log_every": 10,
                "portfolio_class": portfolio,
                "portfolio_kwargs": {
                    "max_weight": 0.04,
                    "individual_turnover": 0.02,
                    "total_turnover": 0.4,
                    "alpha" : 100,
                    "alpha_on": True,
                    "hard_turnover": 0.3,
                },
                "save_filename": results_folder / "results_alpha_h_base_hard.npz"

            }
        ),
        "alpha_h_full_hard": (
            loop_optimization,
            {
                "start_day" : "2015-07-01",
                "end_day" : "2026-07-01",
                "log_every": 100,
                "portfolio_class": portfolio,
                "portfolio_kwargs": {
                    "max_weight": 0.04,
                    "individual_turnover": 0.02,
                    "total_turnover": 0.4,
                    "alpha" : 100,
                    "alpha_on": True,
                    "hard_turnover": 0.3,
                },
                "save_filename": results_folder / "results_alpha_h_full_hard.npz"

            }
        ),
        "alpha_i_full_hard": (
            loop_optimization,
            {
                "start_day" : "2015-07-01",
                "end_day" : "2026-07-01",
                "log_every": 100,
                "portfolio_class": portfolio,
                "portfolio_kwargs": {
                    "max_weight": 0.04,
                    "individual_turnover": 0.02,
                    "total_turnover": 0.4,
                    "alpha" : 10,
                    "alpha_on": True,
                    "hard_turnover": 0.3,
                },
                "save_filename": results_folder / "results_alpha_i_full_hard.npz"

            }
        ),
        "alpha_j_full_hard": (
            loop_optimization,
            {
                "start_day" : "2015-07-01",
                "end_day" : "2026-07-01",
                "log_every": 100,
                "portfolio_class": portfolio,
                "portfolio_kwargs": {
                    "max_weight": 0.04,
                    "individual_turnover": 0.02,
                    "total_turnover": 0.4,
                    "alpha" : 0.5,
                    "alpha_on": True,
                    "hard_turnover": 0.3,
                },
                "save_filename": results_folder / "results_alpha_j_full_hard.npz"

            }
        ),
        "alpha_k_full_hard": (
            loop_optimization,
            {
                "start_day" : "2015-07-01",
                "end_day" : "2026-07-01",
                "log_every": 100,
                "portfolio_class": portfolio,
                "portfolio_kwargs": {
                    "max_weight": 0.04,
                    "individual_turnover": 0.02,
                    "total_turnover": 0.4,
                    "alpha" : 0.05,
                    "alpha_on": True,
                    "hard_turnover": 0.3,
                },
                "save_filename": results_folder / "results_alpha_k_full_hard.npz"

            }
        ),
        "alpha_l_full_hard": (
            loop_optimization,
            {
                "start_day" : "2015-07-01",
                "end_day" : "2026-07-01",
                "log_every": 100,
                "portfolio_class": portfolio,
                "portfolio_kwargs": {
                    "max_weight": 0.04,
                    "individual_turnover": 0.02,
                    "total_turnover": 0.4,
                    "alpha" : 5,
                    "alpha_on": True,
                    "hard_turnover": 0.3,
                },
                "save_filename": results_folder / "results_alpha_l_full_hard.npz"

            }
        ),
        "alpha_m_full_hard": (
            loop_optimization,
            {
                "start_day" : "2015-07-01",
                "end_day" : "2026-07-01",
                "log_every": 100,
                "portfolio_class": portfolio,
                "portfolio_kwargs": {
                    "max_weight": 0.04,
                    "individual_turnover": 0.02,
                    "total_turnover": 0.4,
                    "alpha" : 1000,
                    "alpha_on": True,
                    "hard_turnover": 0.3,
                },
                "save_filename": results_folder / "results_alpha_m_full_hard.npz"

            }
        ),
        "alpha_n_full_hard": (
            loop_optimization,
            {
                "start_day" : "2015-07-01",
                "end_day" : "2026-07-01",
                "log_every": 100,
                "portfolio_class": portfolio,
                "portfolio_kwargs": {
                    "max_weight": 0.04,
                    "individual_turnover": 0.02,
                    "total_turnover": 0.4,
                    "alpha" : 10000,
                    "alpha_on": True,
                    "hard_turnover": 0.3,
                },
                "save_filename": results_folder / "results_alpha_n_full_hard.npz"

            }
        ),


        # ------------------
        # ------------------
        # ELASTIC NET RUNS
        # ------------------
        # ------------------

        # ------------------
        # ------------------
        # BASELINE RUNS
        # ------------------
        # ------------------
        "elastic_a_base": (
            loop_optimization,
            {
                "start_day" : "2015-07-01",
                "end_day" : "2015-07-21",
                "log_every": 10,
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "max_weight": 0.04,
                    "individual_turnover": 0.02,
                    "total_turnover": 0.7,
                    "lambda_a" : 10,
                    "lambda_b" : 0.02,
                    "alpha_on": False,
                },

            }
        ),
        "elastic_b_base": (
            loop_optimization,
            {
                "start_day" : "2015-07-01",
                "end_day" : "2015-07-21",
                "log_every": 10,
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "total_turnover": 0.7,
                    "lambda_a" : 0.1,
                    "lambda_b" : 0.1,
                    "alpha_on": False,
                },
            }
        ),
        "elastic_c_base": (
            loop_optimization,
            {
                "start_day" : "2015-07-01",
                "end_day" : "2015-07-21",
                "log_every": 10,
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.0001,
                    "lambda_b" : 0.1,
                },

            }
        ),
        "elastic_d_base": (
            loop_optimization,
            {
                "start_day" : "2015-07-01",
                "end_day" : "2015-07-21",
                "log_every": 10,
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.03,
                    "lambda_b" : 0.5,
                },

            }
        ),
        "elastic_e_base": (
            loop_optimization,
            {
                "start_day" : "2015-07-01",
                "end_day" : "2015-07-21",
                "log_every": 10,
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.04,
                    "lambda_b" : 0.5,
                },

            }
        ),
        "elastic_f_base": (
            loop_optimization,
            {
                "start_day" : "2015-07-01",
                "end_day" : "2015-07-21",
                "log_every": 10,
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0,
                    "lambda_b" : 1,
                },

            }
        ),
        "elastic_g_base": (
            loop_optimization,
            {
                "start_day" : "2015-07-01",
                "end_day" : "2015-07-21",
                "log_every": 10,
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.00001,
                    "lambda_b" : 0.5,
                },

            }
        ),
        "elastic_h_base": (
            loop_optimization,
            {
                "start_day" : "2015-07-01",
                "end_day" : "2015-07-21",
                "log_every": 10,
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.0001,
                    "lambda_b" : 0.5,
                },

            }
        ),
        # ---------------
        # --------------
        # ELASTIC NET WITH LB =  0.1
        # --------------
        # --------------
        "elastic_a_full_1": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.005,
                    "lambda_b" : 0.1,
                },
            }
        ),
        "elastic_b_full_1": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.01,
                    "lambda_b" : 0.1,
                },
            }
        ),
        "elastic_c_full_1": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.05,
                    "lambda_b" : 0.1,
                },
            }
        ),
        "elastic_d_full_1": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.1,
                    "lambda_b" : 0.1,
                },
            }
        ),
        "elastic_e_full_1": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.5,
                    "lambda_b" : 0.1,
                },
            }
        ),
        "elastic_f_full_1": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 1,
                    "lambda_b" : 0.1,
                },
            }
        ),
        "elastic_g_full_1": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 10,
                    "lambda_b" : 0.1,
                },
            }
        ),

        # ---------------
        # --------------
        # ELASTIC NET WITH LB =  0.5
        # --------------
        # --------------
        "elastic_a_full_2": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.005,
                    "lambda_b" : 0.5,
                },
            }
        ),
        "elastic_b_full_2": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.01,
                    "lambda_b" : 0.5,
                },
            }
        ),
        "elastic_c_full_2": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.05,
                    "lambda_b" : 0.5,
                },
            }
        ),
        "elastic_d_full_2": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.1,
                    "lambda_b" : 0.5,
                },
            }
        ),
        "elastic_e_full_2": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.5,
                    "lambda_b" : 0.5,
                },
            }
        ),
        "elastic_f_full_2": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 1,
                    "lambda_b" : 0.5,
                },
            }
        ),
        "elastic_g_full_2": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 10,
                    "lambda_b" : 0.5,
                },
            }
        ),
        # ---------------
        # --------------
        # ELASTIC NET WITH LB =  1
        # --------------
        # --------------
        "elastic_a_full_3": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.005,
                    "lambda_b" : 1,
                },
            }
        ),
        "elastic_b_full_3": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.01,
                    "lambda_b" : 1,
                },
            }
        ),
        "elastic_c_full_3": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.05,
                    "lambda_b" : 1,
                },
            }
        ),
        "elastic_d_full_3": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.1,
                    "lambda_b" : 1,
                },
            }
        ),
        "elastic_e_full_3": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 0.5,
                    "lambda_b" : 1,
                },
            }
        ),
        "elastic_f_full_3": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 1,
                    "lambda_b" : 1,
                },
            }
        ),
        "elastic_g_full_3": (
            loop_optimization,
            {
                "portfolio_class": elastic_net,
                "portfolio_kwargs": {
                    "lambda_a" : 10,
                    "lambda_b" : 1,
                },
            }
        ),

    }
    return cases


# import dataclasses
# import typing

# @dataclasses.dataclass
# class Param:
#   alpha: float
#   func: typing.Any
#   pos: float
#   tvr: float
#   gate: float

#   def __call__(self, wc, wt, attr) -> Any:
#      pass

#   def __str__(self) -> str:
#     return f"alpha_{self.alpha:.4f}_{self.pos:.2f}_{self.tvr:.2f}"

#   @property
#   def filename(self) -> str:
#     return f"{self!s}.npz"

# @dataclasses.dataclass
# class BerhuParam (Param):
#     pass

# def run(case):
#     if isinstance(case, BerhuParam):
#         pass
#     elif isinstance(case, Param):
#         pass

# def make_cases():
#   ans = {}
#   for alpha in (1e-3, 1e-2, 1e-1, 1):
#     case = Param(alpha, "some_function", 0.4, 0.3, 0.2)
#     if str(case) in ans: raise ValueError
#     ans[str(case)] = case


#   return ans

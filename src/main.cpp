#include <cairomm/cairomm.h>
#include <gtkmm.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

using namespace std;

double sq(double x) { return x * x; }

double rmse_vec(const vector<double>& a, const vector<double>& b) {
	double s = 0;
	for (size_t i = 0; i < a.size(); i++) s += sq(a[i] - b[i]);
	return sqrt(s / a.size());
}

bool invertMatrix(vector<vector<double>> A, vector<vector<double>>& inv) {
	int n = A.size();
	inv.assign(n, vector<double>(n, 0.0));
	for (int i = 0; i < n; i++) inv[i][i] = 1.0;
	const double EPS = 1e-14;
	for (int i = 0; i < n; i++) {
		double pivot = A[i][i];
		int pivRow = i;
		if (fabs(pivot) < EPS) {
			for (int r = i + 1; r < n; r++)
				if (fabs(A[r][i]) > fabs(pivot)) {
					pivot = A[r][i];
					pivRow = r;
				}
			if (fabs(pivot) < EPS) return false;
			swap(A[i], A[pivRow]);
			swap(inv[i], inv[pivRow]);
		}
		pivot = A[i][i];
		for (int j = 0; j < n; j++) {
			A[i][j] /= pivot;
			inv[i][j] /= pivot;
		}
		for (int r = 0; r < n; r++) {
			if (r == i) continue;
			double f = A[r][i];
			for (int c = 0; c < n; c++) {
				A[r][c] -= f * A[i][c];
				inv[r][c] -= f * inv[i][c];
			}
		}
	}
	return true;
}

vector<double> matVec(const vector<vector<double>>& M,
		      const vector<double>& v) {
	int n = M.size();
	vector<double> out(n, 0.0);
	for (int i = 0; i < n; i++)
		for (size_t j = 0; j < v.size(); j++) out[i] += M[i][j] * v[j];
	return out;
}

void buildNormalEq(const vector<double>& xs, const vector<double>& ys,
		   int degree, vector<vector<double>>& XT_X,
		   vector<double>& XT_y) {
	int D = degree + 1;
	XT_X.assign(D, vector<double>(D, 0.0));
	XT_y.assign(D, 0.0);
	int N = xs.size();
	vector<vector<double>> powx(N, vector<double>(2 * D + 1, 1.0));
	for (int i = 0; i < N; i++)
		for (int p = 1; p < 2 * D + 1; p++)
			powx[i][p] = powx[i][p - 1] * xs[i];
	for (int r = 0; r < D; r++) {
		for (int c = 0; c < D; c++) {
			double s = 0;
			for (int i = 0; i < N; i++) s += powx[i][r + c];
			XT_X[r][c] = s;
		}
		double sy = 0;
		for (int i = 0; i < N; i++) sy += powx[i][r] * ys[i];
		XT_y[r] = sy;
	}
}

bool fit_poly(const vector<double>& xs, const vector<double>& ys, int degree,
	      double ridge, vector<double>& coef_out) {
	vector<vector<double>> XT_X;
	vector<double> XT_y;
	buildNormalEq(xs, ys, degree, XT_X, XT_y);
	for (int i = 0; i <= degree; i++) XT_X[i][i] += ridge;
	vector<vector<double>> inv;
	if (!invertMatrix(XT_X, inv)) return false;
	coef_out = matVec(inv, XT_y);
	return true;
}

double predict_poly(double x, const vector<double>& coef) {
	double sum = 0, xi = 1;
	for (double c : coef) {
		sum += c * xi;
		xi *= x;
	}
	return sum;
}

class GraphArea : public Gtk::DrawingArea {
       public:
	vector<double> X, Y, Y_actual;
	void set_data(const vector<double>& x, const vector<double>& y_pred,
		      const vector<double>& y_actual) {
		X = x;
		Y = y_pred;
		Y_actual = y_actual;
		queue_draw();
	}

	double mouse_x = -1, mouse_y = -1;
	bool mouse_inside = false;

	GraphArea() {
		add_events(Gdk::POINTER_MOTION_MASK | Gdk::LEAVE_NOTIFY_MASK);
	}

       protected:
	bool on_motion_notify_event(GdkEventMotion* e) override {
		mouse_x = e->x;
		mouse_y = e->y;
		mouse_inside = true;
		queue_draw();
		return true;
	}

	bool on_leave_notify_event(GdkEventCrossing*) override {
		mouse_inside = false;
		queue_draw();
		return true;
	}

	bool on_draw(const Cairo::RefPtr<Cairo::Context>& cr) override {
		int W = get_width(), H = get_height();

		cr->begin_new_sub_path();
		cr->arc(20, 20, 20, M_PI, 3 * M_PI / 2);
		cr->arc(W - 20, 20, 20, 3 * M_PI / 2, 2 * M_PI);
		cr->arc(W - 20, H - 20, 20, 0, M_PI / 2);
		cr->arc(20, H - 20, 20, M_PI / 2, M_PI);
		cr->close_path();

		cr->set_source_rgb(0.129, 0.129, 0.129);

		cr->fill_preserve();
		cr->set_source_rgb(1.0, 1.0, 1.0);

		cr->set_line_width(2);
		cr->stroke();

		if (X.empty()) return true;

		double xmin = *min_element(X.begin(), X.end()),
		       xmax = *max_element(X.begin(), X.end());
		double ymin =
		    min(*min_element(Y_actual.begin(), Y_actual.end()),
			*min_element(Y.begin(), Y.end()));

		double ymax =
		    max(*max_element(Y_actual.begin(), Y_actual.end()),
			*max_element(Y.begin(), Y.end()));

		if (fabs(ymax - ymin) < 1e-9) ymax = ymin + 1;

		if (fabs(ymax - ymin) < 1e-9) ymax = ymin + 1;

		auto mapx = [&](double v) {
			return 60 + (v - xmin) / (xmax - xmin) * (W - 100);
		};
		auto mapy = [&](double v) {
			return H - 60 - (v - ymin) / (ymax - ymin) * (H - 100);
		};

		auto invx = [&](double px) {
			return xmin + (px - 60) / (W - 100) * (xmax - xmin);
		};
		auto invy = [&](double py) {
			return ymin + (H - 60 - py) / (H - 100) * (ymax - ymin);
		};

		cr->set_source_rgb(0.878, 0.878, 0.878);
		cr->set_line_width(2);
		cr->move_to(60, H - 60);
		cr->line_to(W - 40, H - 60);
		cr->stroke();
		cr->move_to(60, H - 60);
		cr->line_to(60, 40);
		cr->stroke();

		cr->select_font_face("Sans", Cairo::FONT_SLANT_NORMAL,
				     Cairo::FONT_WEIGHT_NORMAL);
		cr->set_font_size(12);
		for (int i = 0; i <= 5; i++) {
			double xv = xmin + i * (xmax - xmin) / 5;
			cr->move_to(mapx(xv) - 10, H - 45);
			cr->set_source_rgb(0.878, 0.878, 0.878);
			cr->show_text(to_string(xv));
			cr->set_source_rgb(0.878, 0.878, 0.878);
			cr->move_to(mapx(xv), H - 60);
			cr->line_to(mapx(xv), H - 55);
			cr->stroke();
		}
		for (int i = 0; i <= 5; i++) {
			double yv = ymin + i * (ymax - ymin) / 5;
			cr->move_to(40, mapy(yv) + 5);
			cr->set_source_rgb(0.878, 0.878, 0.878);
			cr->show_text(to_string(yv));
			cr->set_source_rgb(0.878, 0.878, 0.878);
			cr->move_to(60, mapy(yv));
			cr->line_to(65, mapy(yv));
			cr->stroke();
		}

		cr->set_source_rgb(0.902, 0.298, 0.235);

		for (size_t i = 0; i < X.size(); i++) {
			cr->arc(mapx(X[i]), mapy(Y_actual[i]), 5, 0, 2 * M_PI);
			cr->fill();
		}

		cr->set_source_rgb(0.251, 0.788, 0.902);

		cr->set_line_width(2);
		for (size_t i = 1; i < X.size(); i++) {
			cr->move_to(mapx(X[i - 1]), mapy(Y[i - 1]));
			cr->line_to(mapx(X[i]), mapy(Y[i]));
			cr->stroke();
		}

		cr->set_source_rgba(0.902, 0.298, 0.235, 0.35);

		size_t pred_start = Y_actual.size();

		cr->move_to(mapx(X[pred_start]), mapy(ymin));

		for (size_t i = pred_start; i < X.size(); i++) {
			cr->line_to(mapx(X[i]), mapy(Y[i]));
		}

		cr->line_to(mapx(X.back()), mapy(ymin));
		cr->close_path();
		cr->fill();

		if (mouse_inside && mouse_x > 60 && mouse_x < W - 40 &&
		    mouse_y > 40 && mouse_y < H - 60) {
			double gx = invx(mouse_x);
			double gy = invy(mouse_y);

			double bw = 160;
			double bh = 50;

			double bx = mouse_x + 12;
			double by = mouse_y + 12;

			if (bx + bw > W - 40) {
				bx = mouse_x - bw - 12;
			}

			if (by + bh > H - 60) {
				by = mouse_y - bh - 12;
			}

			cr->set_source_rgba(0, 0, 0, 0.75);
			cr->rectangle(bx, by, bw, bh);
			cr->fill();

			cr->set_source_rgb(1, 1, 1);
			cr->set_line_width(1);
			cr->rectangle(bx, by, bw, bh);
			cr->stroke();

			cr->set_font_size(12);
			cr->move_to(bx + 8, by + 18);
			cr->show_text("x: " + to_string(gx));

			cr->move_to(bx + 8, by + 34);
			cr->show_text("y: " + to_string(gy));
		}

		return true;
	}
};

class RoundedFrame : public Gtk::Bin {
       private:
	int radius = 20;

       protected:
	bool on_draw(const Cairo::RefPtr<Cairo::Context>& cr) override {
		int width = get_allocated_width();
		int height = get_allocated_height();

		cr->save();
		cr->set_source_rgba(0, 0, 0, 0.3);
		cr->translate(4, 4);
		cr->begin_new_sub_path();
		cr->arc(radius, radius, radius, M_PI, 3 * M_PI / 2);
		cr->arc(width - radius, radius, radius, 3 * M_PI / 2, 2 * M_PI);
		cr->arc(width - radius, height - radius, radius, 0, M_PI / 2);
		cr->arc(radius, height - radius, radius, M_PI / 2, M_PI);
		cr->close_path();
		cr->fill();
		cr->restore();

		cr->begin_new_sub_path();
		cr->arc(radius, radius, radius, M_PI, 3 * M_PI / 2);
		cr->arc(width - radius, radius, radius, 3 * M_PI / 2, 2 * M_PI);
		cr->arc(width - radius, height - radius, radius, 0, M_PI / 2);
		cr->arc(radius, height - radius, radius, M_PI / 2, M_PI);
		cr->close_path();

		cr->set_source_rgb(0.294, 0.294, 0.294);

		cr->fill_preserve();
		cr->set_source_rgb(1.0, 1.0, 1.0);

		cr->set_line_width(2);
		cr->stroke();

		return Gtk::Bin::on_draw(cr);
	}

	void on_size_allocate(Gtk::Allocation& allocation) override {
		set_allocation(allocation);
		if (get_child()) get_child()->size_allocate(allocation);
	}
};

class RoundedTextView : public Gtk::TextView {
       private:
	int radius = 20;

       protected:
	bool on_draw(const Cairo::RefPtr<Cairo::Context>& cr) override {
		int width = get_allocated_width();
		int height = get_allocated_height();

		cr->begin_new_sub_path();
		cr->arc(radius, radius, radius, M_PI, 3 * M_PI / 2);
		cr->arc(width - radius, radius, radius, 3 * M_PI / 2, 2 * M_PI);
		cr->arc(width - radius, height - radius, radius, 0, M_PI / 2);
		cr->arc(radius, height - radius, radius, M_PI / 2, M_PI);
		cr->close_path();

		cr->set_source_rgb(0.180, 0.180, 0.180);

		cr->fill_preserve();
		cr->set_source_rgb(1.0, 1.0, 1.0);

		cr->set_line_width(2);
		cr->stroke();

		return Gtk::TextView::on_draw(cr);
	}

       public:
	RoundedTextView() {
		auto css_provider = Gtk::CssProvider::create();
		css_provider->load_from_data(R"(
            GtkTextView {
                border: 2px solid white;
                border-radius: 20px;
                background-color: #2d2d2d;
                color: #e0e0e0;
            }
        )");

		auto screen = Gdk::Screen::get_default();
		auto style_context = get_style_context();
		style_context->add_provider_for_screen(
		    screen, css_provider,
		    GTK_STYLE_PROVIDER_PRIORITY_APPLICATION + 1);
	}
};

class RoundedButton : public Gtk::Button {
       private:
	int radius = 20;

       protected:
	bool on_draw(const Cairo::RefPtr<Cairo::Context>& cr) override {
		int width = get_allocated_width();
		int height = get_allocated_height();

		cr->begin_new_sub_path();
		cr->arc(radius, radius, radius, M_PI, 3 * M_PI / 2);
		cr->arc(width - radius, radius, radius, 3 * M_PI / 2, 2 * M_PI);
		cr->arc(width - radius, height - radius, radius, 0, M_PI / 2);
		cr->arc(radius, height - radius, radius, M_PI / 2, M_PI);
		cr->close_path();

		cr->set_source_rgb(0.2, 0.2, 0.2);

		cr->fill_preserve();
		cr->set_source_rgb(1.0, 1.0, 1.0);

		cr->set_line_width(2);
		cr->stroke();

		if (auto label = dynamic_cast<Gtk::Label*>(get_child())) {
			cr->set_source_rgb(0.9, 0.9, 0.9);
			cr->select_font_face("Sans", Cairo::FONT_SLANT_NORMAL,
					     Cairo::FONT_WEIGHT_BOLD);
			cr->set_font_size(14);

			Cairo::TextExtents extents;
			cr->get_text_extents(label->get_label(), extents);

			double x =
			    (width - extents.width) / 2 - extents.x_bearing;
			double y =
			    (height - extents.height) / 2 - extents.y_bearing;

			cr->move_to(x, y);
			cr->show_text(label->get_label());
		}

		return true;
	}

       public:
	RoundedButton(const std::string& label) : Gtk::Button(label) {}
};

class MainWindow : public Gtk::Window {
       public:
	Gtk::Box mainBox{Gtk::ORIENTATION_HORIZONTAL, 20};
	Gtk::Box leftVBox{Gtk::ORIENTATION_VERTICAL, 15};
	RoundedTextView inputBox;
	RoundedButton plotBtn;
	GraphArea graph;

	MainWindow() : plotBtn("Plot") {
		set_default_size(950, 600);
		set_title("FuturePoint");

		auto screen = Gdk::Screen::get_default();
		auto css_provider = Gtk::CssProvider::create();
		css_provider->load_from_data(R"(
            .background {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            GtkWindow {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            GtkTextBuffer {
                background-color: #2d2d2d;
                color: #e0e0e0;
            }
            GtkLabel {
                color: #e0e0e0;
            }
        )");

		auto style_context = get_style_context();
		style_context->add_provider_for_screen(
		    screen, css_provider,
		    GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);

		auto roundedContainer = Gtk::make_managed<RoundedFrame>();
		roundedContainer->add(leftVBox);

		add(mainBox);
		mainBox.pack_start(*roundedContainer, Gtk::PACK_SHRINK);
		mainBox.pack_start(graph, Gtk::PACK_EXPAND_WIDGET);

		mainBox.set_margin_left(20);
		mainBox.set_margin_right(20);
		mainBox.set_margin_top(20);
		mainBox.set_margin_bottom(20);

		leftVBox.set_margin_left(20);
		leftVBox.set_margin_right(20);
		leftVBox.set_margin_top(20);
		leftVBox.set_margin_bottom(20);

		inputBox.set_size_request(240, 450);
		inputBox.override_background_color(Gdk::RGBA("#2d2d2d"));
		inputBox.override_color(Gdk::RGBA("#e0e0e0"));
		leftVBox.pack_start(inputBox, Gtk::PACK_SHRINK);
		leftVBox.pack_start(plotBtn, Gtk::PACK_SHRINK);

		plotBtn.signal_clicked().connect(
		    sigc::mem_fun(*this, &MainWindow::on_plot));

		show_all_children();
	}

	void on_plot() {
		vector<double> x, y;
		auto buffer = inputBox.get_buffer();
		Glib::ustring text = buffer->get_text();
		istringstream iss(text.raw());
		string line;
		while (getline(iss, line)) {
			istringstream ls(line);
			double a, b;
			if (ls >> a >> b) {
				x.push_back(a);
				y.push_back(b);
			}
		}
		if (x.size() < 2) return;

		vector<double> coef;
		fit_poly(x, y, 2, 1e-3, coef);

		vector<double> Xpred = x;
		vector<double> Ypred;

		double step = x.back() - x[x.size() - 2];

		for (double xv : x) Ypred.push_back(predict_poly(xv, coef));

		double lastX = x.back();
		for (int i = 1; i <= 5; i++) {
			double xv = lastX + i * step;
			Xpred.push_back(xv);
			Ypred.push_back(predict_poly(xv, coef));
		}

		graph.set_data(Xpred, Ypred, y);
	}
};

int main(int argc, char** argv) {
	auto app =
	    Gtk::Application::create(argc, argv, "com.futurepoint.app");
	MainWindow win;
	return app->run(win);
}

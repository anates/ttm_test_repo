/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2012 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Sven Wetterauer, University of Heidelberg, 2012
 */


// @sect3{Include files}

// The first few files have already been covered in previous examples and will
// thus not be further commented on.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>

#include <Sacado.hpp>
#include <boost/math/special_functions/factorials.hpp>

//namespace LA
//{
//	using namespace ::LinearAlgebraTrilinos;
//}

#include <fstream>
#include <iostream>
#include <iomanip>
#include <experimental/type_traits>

//#define PULSE_LENGTH 1.5e-13
//#define PULSE_ENERGY 1.7e-6
//#define INIT_BEAM_RAD 1e-3
//#define TARGET_BEAM_RAD 1e-6
//#define PULSE_INTENSITY (INIT_BEAM_RAD*INIT_BEAM_RAD)/(TARGET_BEAM_RAD*TARGET_BEAM_RAD)*PULSE_ENERGY/PULSE_LENGTH
#define INNER_CIRCLE_RADIUS 5e-2

template < typename T >
using toDim_t = decltype(std::declval<T>().norm());

template < typename T >
using has_toDim = std::experimental::is_detected< toDim_t, T >;

template <bool, typename T = void>
struct enable_if
{};

template <typename T>
struct enable_if<true, T> {
		typedef T type;
};

// We will use adaptive mesh refinement between Newton iterations. To do so,
// we need to be able to work with a solution on the new mesh, although it was
// computed on the old one. The SolutionTransfer class transfers the solution
// from the old to the new mesh:


// We then open a namespace for this program and import everything from the
// dealii namespace into it, as in previous programs:
namespace Step15
{
	using namespace dealii;


	// @sect3{The <code>MinimalSurfaceProblem</code> class template}

	// The class template is basically the same as in step-6.  Three additions
	// are made:
	// - There are two solution vectors, one for the Newton update
	//   $\delta u^n$, and one for the current iterate $u^n$.
	// - The <code>setup_system</code> function takes an argument that denotes whether
	//   this is the first time it is called or not. The difference is that the
	//   first time around we need to distribute the degrees of freedom and set the
	//   solution vector for $u^n$ to the correct size. The following times, the
	//   function is called after we have already done these steps as part of
	//   refining the mesh in <code>refine_mesh</code>.
	// - We then also need new functions: <code>set_boundary_values()</code>
	//   takes care of setting the boundary values on the solution vector
	//   correctly, as discussed at the end of the
	//   introduction. <code>compute_residual()</code> is a function that computes
	//   the norm of the nonlinear (discrete) residual. We use this function to
	//   monitor convergence of the Newton iteration. The function takes a step
	//   length $\alpha^n$ as argument to compute the residual of $u^n + \alpha^n
	//   \; \delta u^n$. This is something one typically needs for step length
	//   control, although we will not use this feature here. Finally,
	//   <code>determine_step_length()</code> computes the step length $\alpha^n$
	//   in each Newton iteration. As discussed in the introduction, we here use a
	//   fixed step length and leave implementing a better strategy as an
	//   exercise.

	void print_status_update(const ConditionalOStream &pcout, const std::string input, const bool do_it = false)
	{
#define PRINT_DEBUG
#ifdef PRINT_DEBUG
		if(do_it)
		{
			pcout << input << '\n';
			//getchar();
		}
#endif
	}

	enum equation_class{carrier_density, electron_temperature, lattice_temperature};



	template<int dim>
	class physics_equations
	{
			//friend class left_right_equations;
		public://Equations
			physics_equations(const double wavelength);
			~physics_equations();

			template <typename T>
			T gamma_func(const typename enable_if<has_toDim<T>::value, T>::type &N_val) const;

			template <typename T>
			T gamma_func(const T &N_val) const;

			template <typename T>
			T k_N(const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const;

			template <typename T>
			T k_N(const T &TL_val) const;

			template <typename T>
			T k_L(const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const;

			template <typename T>
			T k_L(const T &TL_val) const;

			template <typename T>
			T k_E(const typename enable_if<has_toDim<T>::value, T>::type &N_val, const typename enable_if<has_toDim<T>::value, T>::type &TE_val, const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const;

			template <typename T>
			T k_E(const T &N_val, const T &TE_val, const T &TL_val) const;

			template <typename T>
			T mu_e(const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const;

			template <typename T>
			T mu_e(const T &TL_val) const;

			template <typename T>
			T mu_h(const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const;

			template <typename T>
			T mu_h(const T TL_val) const;

			template <typename T>
			T E_g(const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const;

			template <typename T>
			T E_g(const T &TL_val) const;

			template <typename T>
			T dEdT(const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const;

			template <typename T>
			T dEdT(const T &TL_val) const;

			template <typename T>
			T free_carrier_cross_section(const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const;

			template <typename T>
			T free_carrier_cross_section(const T &TL_val) const;

			template <typename T>
			T reflectivity(const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const;

			template <typename T>
			T reflectivity(const T &TL_val) const;

			template <typename T>
			T impact_ionization_coefficient(const typename enable_if<has_toDim<T>::value, T>::type &TE_val, const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const;

			//			template <typename T>
			//			T impact_ionization_coefficient(const T &TE_val, const T &TL_val) const;

			template <typename T, typename U>
			T impact_ionization_coefficient(const T &TE_val, const U &TL_val) const;

			double auger_recombination_coefficient(void) const;

			template <typename T>
			T dNdt_prefactor(const typename enable_if<has_toDim<T>::value, T>::type &TE_val, const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const;

			template <typename T>
			T dNdt_prefactor(const T &TE_val, const T &TL_val) const;

			template <typename T>
			T c_app_Tensor(const T &TL_val) const;

			template <typename T>
			T c_app(const T &TL_val) const;

			template <typename T>
			T c_e(const typename enable_if<has_toDim<T>::value, T>::type &N_val) const;

			template <typename T>
			T c_e(const T &N_val) const;

			template <typename T>
			T heat_capacity(const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const;

			template <typename T>
			T heat_capacity(const T &TL_val) const;

			template <typename T>
			T integrate_heat_capacity(const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const;

			template <typename T>
			T integrate_heat_capacity(const T &TL_val) const;

			template <typename T>
			T beta_f_func(const T x, const unsigned int n) const;

			template <typename T>
			T one_photon_absorption(const T lattice_temperature) const;//For later use of different wavelengths;

			template <typename T>
			T two_photon_absorption(const T lattice_temperature) const;//For later use of different wavelengths;

			template <typename T>
			T three_photon_absorption(const T lattice_temperature) const;//For later use of different wavelengths;

			double photon_energy_func() const;

			//private://constants
			const double molar_mass_volume = 12.058832e-6;
			const double wavelength;
			const double density = 2.33e3;
			const double refractive_index = 3.52;
			const double hbar_val = 1.0545718e-34;
			const double carrier_relaxation_time = 240e-15;
			const double carrier_density = 6e26;
			const double melting_temperature = 1687;
			const double ambient_temperature = 293;
			const double latent_heat = 48.31e3/molar_mass_volume;
			const double one_photon_absorption_coefficient = 0;
			const double two_photon_absorption_coefficient = 2e-11;
			const double three_photon_absorption_coefficient = 0;
			const double boltzmann_constant = 1.38064852e-23;
			const double electron_charge = 1.60217662e-19;
			const double planck_constant = 6.626069934e-34;
			const double speed_of_light = 299792458;
			const double w0_val = 2*M_PI*speed_of_light/wavelength;
			const double initial_carrier_density = 9e9*1e6;
			const double ambient_temperature_II = ambient_temperature;


			const double temperature_delta_T = 50;//TBFixed
	};

	template<int dim>
	physics_equations<dim>::physics_equations(const double wavelength) : wavelength(wavelength)
	{

	}

	template<int dim>
	physics_equations<dim>::~physics_equations()
	{

	}

	template<int dim> template <typename T>
	T physics_equations<dim>::gamma_func(const typename enable_if<has_toDim<T>::value, T>::type &N_val) const
	{
		T return_value;
		for(size_t i = 0; i < dim; ++i)
			return_value[i] = (3*this->boltzmann_constant*N_val[i])/(this->carrier_relaxation_time*(1+pow(N_val[i]/this->carrier_density, 2)));
		return return_value;
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::gamma_func(const T &N_val) const
	{
		return (3*this->boltzmann_constant*N_val)/(this->carrier_relaxation_time*(1+pow(N_val/this->carrier_density, 2)));
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::k_N(const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const
	{
		T return_value;
		for(size_t i = 0; i < dim; ++i)
			return_value[i] = 2*this->boltzmann_constant*TL_val[i]/this->electron_charge*(this->mu_e(TL_val[i])*this->mu_h(TL_val[i])/(this->mu_e(TL_val[i])+this->mu_h(TL_val[i])));
		return return_value;
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::k_N(const T &TL_val) const
	{
		return 2*this->boltzmann_constant*TL_val/this->electron_charge*(this->mu_e(TL_val)*this->mu_h<T>(TL_val)/(this->mu_e(TL_val)+this->mu_h<T>(TL_val)));
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::k_L(const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const
	{
		throw("Should not be used\n");
		return 0*TL_val;//TBD
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::k_L(const T &TL_val) const
	{
		return 1/(1.56e-3 * TL_val + 1.65e-6 * pow(TL_val, 2) + 0.03) * 1e2;
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::k_E(const typename enable_if<has_toDim<T>::value, T>::type &N_val, const typename enable_if<has_toDim<T>::value, T>::type &TE_val, const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const
	{
		T return_value;
		for(size_t i = 0; i < dim; ++i)
			return_value[i] = -2*this->boltzmann_constant*this->boltzmann_constant*TE_val[i]*N_val[i]*(this->mu_e(TL_val[i])+this->mu_h(TL_val[i]))/this->electron_charge;
		return return_value;
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::k_E(const T &N_val, const T &TE_val, const T &TL_val) const
	{
		return 2*this->boltzmann_constant*this->boltzmann_constant*TE_val*N_val*(this->mu_e<T>(TL_val)+this->mu_h<T>(TL_val))/this->electron_charge;//Missing the gradient of N, but that is zero atm, so I do not care
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::mu_e(const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const
	{
		T temp_variable;
		for(size_t i = 0; i < dim; ++i)
			temp_variable[i] = 1.35e-1*pow(TL_val[i]/300, -2.4);
		return temp_variable;
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::mu_e(const T &TL_val) const
	{
		return 1.35e-1*pow(TL_val/300, -2.4);
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::mu_h(const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const
	{
		T return_value;
		for(size_t i = 0; i < dim; ++i)
			return_value[i] = 4.8e-2*pow(TL_val[i]/300, -2.5);
		return return_value;
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::mu_h(const T TL_val) const
	{
		return 4.8e-2*pow(TL_val/300, -2.5);
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::E_g(const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const
	{
		T return_value;
		for(size_t i = 0; i < dim; ++i)
			return_value[i] = (1.557-7.021e-4*pow(TL_val[i], 2)/(TL_val[i]+1108))*electron_charge;
		return return_value;
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::E_g(const T &TL_val) const
	{
		return (1.557-7.021e-4*pow(TL_val, 2)/(TL_val+1108))*electron_charge;
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::dEdT(const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const
	{
		T return_value;
		for(size_t i = 0; i < dim; ++i)
			return_value[i] = -7.021e-4*(pow(TL_val[i], 2)+2216*TL_val[i])/pow(TL_val[i]+1108, 2);
		return return_value;
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::dEdT(const T &TL_val) const
	{
		return -7.021e-4*(pow(TL_val, 2)+2216*TL_val)/pow(TL_val+1108, 2);
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::free_carrier_cross_section(const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const
	{
		return 1.7e-24*TL_val;
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::free_carrier_cross_section(const T &TL_val) const
	{
		return 1.7e-24*TL_val;
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::reflectivity(const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const
	{
		return 0.3+5e-5*(TL_val-300);
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::reflectivity(const T &TL_val) const
	{
		return 0.3+5e-5*(TL_val-300);
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::impact_ionization_coefficient(const typename enable_if<has_toDim<T>::value, T>::type &TE_val, const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const
	{
		T bandgap_energy = E_g(TL_val);
		T return_value;
		for(size_t i = 0; i < dim; ++i)
			return_value[i] = 3.6e10*exp(-1.5*bandgap_energy[i]/(boltzmann_constant*TE_val[i]));
		return return_value;
	}

	//	template<int dim> template <typename T>
	//	T physics_equations<dim>::impact_ionization_coefficient(const T &TE_val, const T &TL_val) const
	//	{
	//		T bandgap_energy = E_g(TL_val);
	//		return 3.6e10*exp(-1.5*bandgap_energy/(boltzmann_constant*TE_val));
	//	}

	template<int dim> template <typename T, typename U>
	T physics_equations<dim>::impact_ionization_coefficient(const T &TE_val, const U &TL_val) const
	{
		U bandgap_energy = E_g(TL_val);
		return 3.6e10*exp(-1.5*bandgap_energy/(boltzmann_constant * TE_val));
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::dNdt_prefactor(const typename enable_if<has_toDim<T>::value, T>::type &TE_val, const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const
	{
		return this->E_g(TL_val)+this->boltzmann_constant*TE_val*3;
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::dNdt_prefactor(const T &TE_val, const T &TL_val) const
	{
		return this->E_g(TL_val)+this->boltzmann_constant*TE_val*3;
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::heat_capacity(const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const
	{
		T return_value;
		for(size_t i = 0; i < dim; ++i)
			return_value[i] = 22.81719 + 3.899510 * (TL_val[i]/1000) - 0.082885 * pow(TL_val[i]/1000, 2) + 0.042111 * pow(TL_val[i]/1000, 3) - 0.354063 * pow(TL_val[i]/1000, -2); //(NIST)//24.5+1.5e3*TL_val[i]/1000-4.37e-5*pow(TL_val[i]/1000, -2);
		return return_value;
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::heat_capacity(const T &TL_val) const
	{
		T local_T_val = TL_val / 1;
		return 22.81719 + 3.899510 * local_T_val - 0.082885 * pow(local_T_val, 2) + 0.042111 * pow(local_T_val, 3) - 0.354063 * pow(local_T_val, -2); //(NIST)
		//24.5+1.5*TL_val/1000-4.37*pow(TL_val/1000, -2);//Temporary fix
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::integrate_heat_capacity(const typename enable_if<has_toDim<T>::value, T>::type &TL_val) const
	{
		double T_lower = this->melting_temperature - this->temperature_delta_T;
		double T_upper = this->melting_temperature + this->temperature_delta_T;
		double T_tmp = 0*TL_val*T_lower*T_upper;//TBD
		(void) T_tmp;
		return 28.9;//
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::integrate_heat_capacity(const T &TL_val) const
	{
		double T_lower = this->melting_temperature - this->temperature_delta_T;
		double T_upper = this->melting_temperature + this->temperature_delta_T;
		(void) T_lower;
		(void) T_upper;
		(void) TL_val;
		//return 0*TL_val*T_lower*T_upper;//TBD
		return 28.9;//TBD
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::c_app_Tensor(const T &TL_val) const
	{
		double T_lower = this->melting_temperature - this->temperature_delta_T;
		double T_upper = this->melting_temperature + this->temperature_delta_T;

		T return_value;
		for(size_t i = 0; i < dim; ++i)
		{
			if(TL_val[i] < T_upper && TL_val[i] > T_lower)
				return_value[i] = this->integrate_heat_capacity(TL_val[i]);
			else
				return_value[i] = this->heat_capacity(TL_val[i]);
		}
		return 0;//return_value;
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::c_app(const T &TL_val) const
	{
		T T_lower = this->melting_temperature - this->temperature_delta_T;
		T T_upper = this->melting_temperature + this->temperature_delta_T;
		(void) T_lower;
		(void) T_upper;
		if(TL_val < T_upper && TL_val > T_lower)
			return this->integrate_heat_capacity(TL_val)/molar_mass_volume;
		else
			return this->heat_capacity(TL_val)/molar_mass_volume;
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::c_e(const typename enable_if<has_toDim<T>::value, T>::type &N_val) const
	{
		return boltzmann_constant*N_val*3;
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::c_e(const T &N_val) const
	{
		return boltzmann_constant*N_val*3;
	}

	template <typename T>
	T alpha_1(const T energy)
	{
		return 0.504 * sqrt(energy) + 392 * pow(energy - 0.0055, 2);
	}

	template <typename T>
	T alpha_2(const T energy)
	{
		return 18.08 * sqrt(energy) + 5760 * pow(energy - 0.0055, 2);
	}

	template<int dim> template<typename T>
	T physics_equations<dim>::one_photon_absorption(const T lattice_temperature) const
	{
//		const double photon_energy = photon_energy_func() / electron_charge;
//		//const double alpha_1 = 0.504 * sqrt(photon_energy) + 0.068 * pow(photon_energy - 0.0055, 2);
//		const double theta_1 = 212;
//		//const double alpha_2 = 18.08 * sqrt(photon_energy) + 1 * pow(photon_energy - 0.0055, 2);
//		const double theta_2 = 670;
//		//const double alpha[] = {alpha_1, alpha_2};
//		const double theta[] = {theta_1, theta_2};
//		T bandgap_energy = E_g(lattice_temperature) / electron_charge;
//		//std::cout << "Alpha_1 is " << alpha_1 << " and alpha_2 is " << alpha_2 << '\n';
//		std::cout << "Bandgap energy is " << bandgap_energy << '\n';
//		T absorption_coefficient = 0;

//		for(size_t i = 1; i <= 2; ++i)
//			for(size_t l = 1; l <= 2; ++l)
//			{
//				T alpha_val = 0;
//				if(i == 1)
//					alpha_val = alpha_1(photon_energy - bandgap_energy + pow(-1, l) * theta[i-1]);
//				else
//					alpha_val = alpha_2(photon_energy - bandgap_energy + pow(-1, l) * theta[i-1]);
//				//std::cout << "Absorption coefficient (" << i << ", " << l << ") is " << pow(-1, l) * (alpha[i-1] * (photon_energy - bandgap_energy + pow(-1, l) * theta[i-1]))/(exp(pow(-1, l) * theta[i-1]/lattice_temperature) - 1) << '\n';
//				std::cout << "Absorption coefficient (" << i << ", " << l << ") is " << pow(-1, l) * (alpha_val)/(exp(pow(-1, l) * theta[i-1]/lattice_temperature) - 1) << '\n';
//			}
		(void) lattice_temperature;
		return one_photon_absorption_coefficient;
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::beta_f_func(const T x, const unsigned int n) const
	{
		return (M_PI*boost::math::double_factorial<double>(2*n+1)/(pow(2, n+2)*boost::math::factorial<double>(n+2)))*pow(2*x, -5)*pow(2*x-1, n+2);
	}

	template<int dim> template <typename T>
	T physics_equations<dim>::two_photon_absorption(const T lattice_temperature) const
	{
		const T E_gi = E_g(lattice_temperature);
		const T beta_f_constant = hbar_val*w0_val/E_gi;
		T beta_val = 2*43e-11*(beta_f_func(beta_f_constant, 0)+beta_f_func(beta_f_constant, 1)+beta_f_func(beta_f_constant, 2));
		return beta_val;
	}

	template<int dim> template<typename T>
	T physics_equations<dim>::three_photon_absorption(const T lattice_temperature) const
	{
		const T E_gi = E_g(lattice_temperature);
		const T gamma_val = 1.54e-3*1e-6*1e-18*pow(E_gi/(hbar_val*w0_val), 9)*pow(hbar_val*w0_val/E_gi-1.0/3.0, 2);
		return gamma_val;
	}

	template<int dim>
	double physics_equations<dim>::photon_energy_func() const
	{
		return planck_constant*speed_of_light/wavelength;
	}

	template<int dim>
	double physics_equations<dim>::auger_recombination_coefficient() const
	{
		return 0;
	}

	template <int dim>
	double pulse_size_prefactor(const Point<dim> p)
	{
		return exp(-(pow(p[0], 2)/(2*pow(INNER_CIRCLE_RADIUS, 2))+pow(p[1], 2)/(2*pow(INNER_CIRCLE_RADIUS, 2))));
	}

	template <int dim, typename T, typename U>
	T full_equation_electron_temperature(const physics_equations<dim> &local_equations, const T &val_TE, const T &val_TE_old, const T &val_TL, const T &val_TL_old, const T &val_N, const T &val_N_old,
										 const T &thermal_source_TE, const T &thermal_source_TE_old, const T &grad_TE, const T &grad_TE_old,
										 const U &fe_values, const U &fe_gradients, const double &time_step, const double &theta)
	{
		return (((val_TE - val_TE_old)/time_step) * fe_values
				+ (theta * pow(val_TE, 2) * grad_TE + (1 - theta) * pow(val_TE_old, 2) * grad_TE_old) * fe_gradients);
	}

	template <typename T>
	T calculate_heat_source_TE(const double wavelength, const double I_val, const T lattice_temperature, const T free_carrier_concentration, const int dim)
	{
		//std::cout << "Intensity is " << I_val << "\n";
		physics_equations<2> local_equations(wavelength);
		(void) dim;
		//        T TL_val = local_equations.ambient_temperature_II;
		//        T return_value = local_equations.impact_ionization_coefficient(electron_temperature, TL_val) * local_equations.carrier_density;
		//        double return_value_II = 0;
		//        return_value_II += local_equations.one_photon_absorption() * I_val/local_equations.photon_energy_func()
		//                + local_equations.two_photon_absorption() * pow(I_val, 2)/(2*local_equations.photon_energy_func())
		//                + local_equations.three_photon_absorption() * pow(I_val, 3)/(3*local_equations.photon_energy_func());
		//        return_value_II -= local_equations.auger_recombination_coefficient() * pow(local_equations.carrier_density, 3);
		//        return return_value + (return_value_II);
		return local_equations.one_photon_absorption(lattice_temperature) * I_val
				+ local_equations.two_photon_absorption(lattice_temperature) * pow(I_val, 2)
				+ local_equations.three_photon_absorption(lattice_temperature) * pow(I_val, 3)
				+ local_equations.free_carrier_cross_section(lattice_temperature) * free_carrier_concentration * I_val;
	}

	double calculate_gaussian_pulse(const double t0, const double pulse_length)
	{
		return exp(-pow(t0, 2)/(2 * pulse_length*pulse_length));
	}

	template <int dim>
	class InitialValues : public Function<dim>
	{
		public:
			InitialValues() : Function<dim>(3 * dim) {}
			virtual double value(const Point<dim> &p, const unsigned int component) const;
			virtual void vector_value(const Point<dim> &p, Vector<double> &values) const;
	};

	template <int dim>
	double InitialValues<dim>::value(const Point<dim> &p, const unsigned int component) const
	{
		(void) p;
		(void) component;
		physics_equations<dim> local_equations(0);
		(void) local_equations;
		if(component < dim)
			return 1;
		else
			if(component < 2 * dim)
				return 2;
			else
				return 3;
	}

	template <int dim>
	void InitialValues<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const
	{
		for(size_t i = 0; i < values.size(); ++i)
			values[i] = value(p, i);
	}

	template <int dim>
	class MinimalSurfaceProblem
	{
		public:
			MinimalSurfaceProblem (const char * file_name);
			~MinimalSurfaceProblem ();

			void run ();

		private:
			void setup_system (const bool initial_step);
			void setup_matrices(const IndexSet &partitioner, const IndexSet &relevant_partitioner);
			void assemble_system (const double Light_intensity, const double Light_intensity_old);
			void solve (const double I_val, const double I_val_old);
			void refine_mesh ();
			void set_boundary_values ();
			double compute_residual (const double alpha, const double Light_intensity, const double Light_intensity_old);
			double determine_step_length ();
			double recalculate_step_length(const double I_val, const double I_val_old);
			void output_results(const int cycle) const;
			void print_usage_message(void);
			void declare_parameters(void);
			void parse_parameters(void);

			MPI_Comm mpi_communicator;

			parallel::distributed::Triangulation<dim>   triangulation;

			const size_t n_components;

			DoFHandler<dim>      dof_handler;
			FESystem<dim>        fe;

			IndexSet locally_owned_dofs;
			IndexSet locally_relevant_dofs;

			ConstraintMatrix     hanging_node_constraints;
			ConstraintMatrix	 newton_constraints;
			ConstraintMatrix	 boundary_constraints;

			LinearAlgebraTrilinos::MPI::SparseMatrix system_matrix;

			LinearAlgebraTrilinos::MPI::Vector      present_solution;
			LinearAlgebraTrilinos::MPI::Vector		old_solution;
			LinearAlgebraTrilinos::MPI::Vector      newton_update;
			LinearAlgebraTrilinos::MPI::Vector      system_rhs;
			LinearAlgebraTrilinos::MPI::Vector		residual;
			LinearAlgebraTrilinos::MPI::Vector		evaluation_point;
			LinearAlgebraTrilinos::MPI::Vector		extension_vector;

			ConditionalOStream pcout;

			std::string log_file_name;
			std::ofstream file_out_stream;

			ParameterHandler         prm;

			/*const*/ size_t max_grid_level;
			/*const*/ size_t min_grid_level;

			/*const*/ size_t max_inner_iterations;
			/*const*/ size_t max_same_res_values;
			double start_time;
			/*const*/ double max_time;
			/*const*/ double time_step;
			/*const*/ double theta;
			double alpha_val;

			double pulse_intensity;
			double pulse_duration;
			double beam_width;
			double focus_width;
			double pulse_wavelength;

			const size_t Free_Carriers = 0;
			const size_t Electron_Temperature = dim;
			const size_t Lattice_Temperature = 2 * dim;
	};

	// @sect3{Boundary condition}


	template <int dim>
	class BoundaryValuesCarriers : public Function<dim>
	{
		public:
			BoundaryValuesCarriers (const size_t n_components) : Function<dim>(dim), n_components(n_components) {}

			virtual double value (const Point<dim>   &p,
								  const unsigned int  component = 0) const;

			virtual void vector_value(const Point<dim> &p, Vector<double> &value) const;
		private:
			const size_t n_components;
	};


	template <int dim>
	double BoundaryValuesCarriers<dim>::value (const Point<dim> &p,
											   const unsigned int component) const
	{
		(void) component;
		(void) p;
		physics_equations<dim> local_equations(0);
		return local_equations.initial_carrier_density;
		//return std::sin(2 * numbers::PI * (p[0]+p[1]));
	}

	template <int dim>
	void BoundaryValuesCarriers<dim>::vector_value(const Point<dim> &p, Vector<double> &value) const
	{
		for(size_t i = 0; i < value.size(); ++i)
			value[i] = BoundaryValuesCarriers<dim>::value(p, i);
	}

	template <int dim>
	class BoundaryValuesTemperatures : public Function<dim>
	{
		public:
			BoundaryValuesTemperatures (const size_t n_components) : Function<dim>(3 * dim), n_components(n_components) {}

			virtual double value (const Point<dim>   &p,
								  const unsigned int  component = 0) const;

			virtual void vector_value(const Point<dim> &p, Vector<double> &value) const;
		private:
			const size_t n_components;
	};


	template <int dim>
	double BoundaryValuesTemperatures<dim>::value (const Point<dim> &p,
												   const unsigned int component) const
	{
		physics_equations<dim> local_equations(0);
		(void) local_equations;
		//return 0;
		return 0;///*std::sin(2 * numbers::PI * (p[0]+p[1])) + */((int)component/(int)2 + 1);
	}

	template <int dim>
	void BoundaryValuesTemperatures<dim>::vector_value(const Point<dim> &p, Vector<double> &value) const
	{
		for(size_t i = 0; i < value.size(); ++i)
			value[i] = BoundaryValuesTemperatures<dim>::value(p, i);
	}

	template <int dim>
	MinimalSurfaceProblem<dim>::MinimalSurfaceProblem (const char *file_name)
		:
		  mpi_communicator(MPI_COMM_WORLD),
		  triangulation(mpi_communicator,
						typename Triangulation<dim>::MeshSmoothing(Triangulation<dim>::smoothing_on_refinement | Triangulation<dim>::smoothing_on_coarsening)),
		  n_components(3),
		  dof_handler (triangulation),
		  fe (FE_Q<dim>(2), n_components * dim),
		  pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
		  alpha_val(0.1)
	{
		this->declare_parameters();
		prm.parse_input(file_name);
		this->parse_parameters();
	}



	template <int dim>
	MinimalSurfaceProblem<dim>::~MinimalSurfaceProblem ()
	{
		file_out_stream.close();
		dof_handler.clear ();
	}

	template <int dim>
	void MinimalSurfaceProblem<dim>::print_usage_message()
	{
		static const char *message
				=
				"\n"
				"TTM-calculator for SI.\n"
				"\n"
				"Usage:\n"
				"    ./main [-p parameter_file]\n"
				"\n"
				"The parameter file has the following format and allows the following\n"
				"values (you can cut and paste this and use it for your own parameter\n"
				"file):\n"
				"\n";
		std::cout << message;
		prm.print_parameters (std::cout, ParameterHandler::Text);
	}

	template <int dim>
	void MinimalSurfaceProblem<dim>::declare_parameters()
	{
		prm.enter_subsection("File processing data");
		{
			prm.declare_entry("Use stdcout", "true", Patterns::Bool(), "If set to true, stdcout will be used, else a file will be used");
			prm.declare_entry("Log file name", "log_file.txt", Patterns::FileName(), "Name of the log file to be used. Old files will be overwritten!");
		}
		prm.leave_subsection();
		prm.enter_subsection("Grid generation data");
		{
			prm.declare_entry("Minimal grid density", "2", Patterns::Integer(), "Value for the minimal grid density");
			prm.declare_entry("Maximal grid density", "8", Patterns::Integer(), "Value for the maximum grid density");
		}
		prm.leave_subsection();
		prm.enter_subsection("Pulse data");
		{
			prm.declare_entry("Maximum pulse energy", "1e6", Patterns::Double());
			prm.declare_entry("Pulse duration", "1e-12", Patterns::Double());
			prm.declare_entry("Pulse focus size", "1e-6", Patterns::Double());
			prm.declare_entry("Original beam radius", "1e-3", Patterns::Double());
			prm.declare_entry("Pulse wavelength", "1.2e-6", Patterns::Double());
		}
		prm.leave_subsection();
		prm.enter_subsection("Simulation parameters");
		{
			prm.declare_entry("Absolute time step", "-1", Patterns::Double(), "If empty, a relative time step can be set");
			prm.declare_entry("Relative time step", "-1", Patterns::Double(), "Time step relative to pulse length");
			prm.declare_entry("Maximum inner iterations", "10", Patterns::Integer());
			prm.declare_entry("Max equal values for residual", "10", Patterns::Integer());
			prm.declare_entry("Theta", "0.5", Patterns::Double());
			prm.declare_entry("Absolute starting time", "-1", Patterns::Double());
			prm.declare_entry("Absolute maximum time", "-1", Patterns::Double());
			prm.declare_entry("Relative starting time", "-1", Patterns::Double());
			prm.declare_entry("Relative maximum time", "-1", Patterns::Double());
		}
		prm.leave_subsection();

	}

	template <int dim>
	void MinimalSurfaceProblem<dim>::parse_parameters()
	{
		prm.enter_subsection("File processing data");
		{
			const bool use_stdcout = prm.get_bool("Use stdcout");
			log_file_name = prm.get("Log file name");
			//			if(use_stdcout)
			//				pcout = ConditionalOStream::ConditionalOStream(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0));
			//			else
			//			{
			//				file_out_stream.open(log_file_name, std::ios::out | std::ios::app);
			//				pcout = ConditionalOStream::ConditionalOStream(file_out_stream, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0));
			//			}
		}
		prm.leave_subsection();
		prm.enter_subsection("Grid generation data");
		{
			min_grid_level = prm.get_integer("Minimal grid density");
			max_grid_level = prm.get_integer("Maximal grid density");
		}
		prm.leave_subsection();
		print_status_update(pcout, std::string("Got minimal and maximal grid level\n"), true);
		prm.enter_subsection("Pulse data");
		{
			const double pulse_energy = prm.get_double("Maximum pulse energy");
			pulse_duration = prm.get_double("Pulse duration");
			focus_width = prm.get_double("Pulse focus size");
			beam_width = prm.get_double("Original beam radius");
			pulse_wavelength = prm.get_double("Pulse wavelength");
			pulse_intensity = pow(beam_width, 2)/pow(focus_width, 2) * pulse_energy/pulse_duration;
		}
		prm.leave_subsection();
		print_status_update(pcout, std::string("Got pulse data\n"), true);
		prm.enter_subsection("Simulation parameters");
		{
			const double abs_time_step = prm.get_double("Absolute time step");
			const double rel_time_step = prm.get_double("Relative time step");
			if(rel_time_step < 0 && abs_time_step < 0)
			{
				throw("Time steps below 0, i.e. no valid values. Stopping\n");
				return;
			}
			else
				if(abs_time_step < 0)
					time_step = rel_time_step * pulse_duration;
				else
					time_step = abs_time_step;
			print_status_update(pcout, std::string("Got time step values\n"), true);
			max_inner_iterations = prm.get_integer("Maximum inner iterations");
			max_same_res_values = prm.get_integer("Max equal values for residual");
			theta = prm.get_double("Theta");
			print_status_update(pcout, std::string("Got theta\n"), true);
			const double abs_start_time = prm.get_double("Absolute starting time");
			const double abs_max_time = prm.get_double("Absolute maximum time");
			const double rel_start_time = prm.get_double("Relative starting time");
			const double rel_max_time = prm.get_double("Relative maximum time");
			if(abs_max_time < 0)
				if(rel_max_time < 0)
				{
					throw("Time data below 0, i.e. now valid values. Stopping\n");
				}
				else
				{
					start_time = rel_start_time * pulse_duration;
					max_time = rel_max_time * pulse_duration;
				}
			else
			{
				start_time = abs_start_time;
				max_time = abs_max_time;
			}
		}
		prm.leave_subsection();
		print_status_update(pcout, std::string("Got simulation data\n"), true);
	}

	// @sect4{MinimalSurfaceProblem::setup_system}

	// As always in the setup-system function, we setup the variables of the
	// finite element method. There are same differences to step-6, because
	// there we start solving the PDE from scratch in every refinement cycle
	// whereas here we need to take the solution from the previous mesh onto the
	// current mesh. Consequently, we can't just reset solution vectors. The
	// argument passed to this function thus indicates whether we can
	// distributed degrees of freedom (plus compute constraints) and set the
	// solution vector to zero or whether this has happened elsewhere already
	// (specifically, in <code>refine_mesh()</code>).

	template <int dim>
	void MinimalSurfaceProblem<dim>::setup_matrices(const IndexSet &partitioner, const IndexSet &relevant_partitioner)
	{
		system_matrix.clear();
		//TrilinosWrappers::SparsityPattern sp(partitioner, partitioner, relevant_partitioner, MPI_COMM_WORLD);
		//DoFTools::make_sparsity_pattern(dof_handler, sp, hanging_node_constraints, false, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
		//DoFTools
		DynamicSparsityPattern dsp(relevant_partitioner);
		DoFTools::make_sparsity_pattern(dof_handler, dsp, boundary_constraints, false);
		SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.n_locally_owned_dofs_per_processor(),
												   mpi_communicator, relevant_partitioner);

		system_matrix.reinit(partitioner, partitioner, dsp, mpi_communicator);
		//		sp.compress();

		//		system_matrix.reinit(sp);
	}


	template <int dim>
	void MinimalSurfaceProblem<dim>::setup_system (const bool initial_step)
	{
		dof_handler.distribute_dofs (fe);
		const size_t dof_numbers = dof_handler.n_dofs();

		std::locale s = pcout.get_stream().getloc();
		pcout.get_stream().imbue(std::locale(""));
		//pcout << std::setprecision(10);
//		std::string output_stream;
//		output_stream = std::string("Number of active cells: ") + std::to_string(triangulation.n_global_active_cells()) + std::string(" (on ") + std::to_string(triangulation.n_levels()) + std::string(" levels)")
//				+ std::string("\n") + std::string("Number of degrees of freedom: ") + std::to_string(dof_numbers) + std::string("\n\n");
//		print_status_update(pcout, output_stream, true);
		pcout.get_stream().imbue(s);

		IndexSet solution_partitioning(dof_numbers), solution_relevant_partitioning(dof_numbers);

		solution_partitioning = dof_handler.locally_owned_dofs();
		DoFTools::extract_locally_relevant_dofs(dof_handler, solution_relevant_partitioning);

		//General constraint matrix
		hanging_node_constraints.clear();
		hanging_node_constraints.reinit(solution_relevant_partitioning);
		DoFTools::make_hanging_node_constraints(dof_handler, hanging_node_constraints);
		hanging_node_constraints.close();
		//Newton constraint matrix;
//		print_status_update(pcout, std::string("Creating newton_constraints\n"), true);
		newton_constraints.clear();
		newton_constraints.reinit(solution_relevant_partitioning);
		DoFTools::make_hanging_node_constraints(dof_handler, newton_constraints);
		VectorTools::interpolate_boundary_values(dof_handler, 0, ZeroFunction<dim>(n_components * dim), newton_constraints);
		newton_constraints.close();

		//Boundary constraints
//		print_status_update(pcout, std::string("Creating boundary_constraints\n"), true);
		boundary_constraints.clear();
		boundary_constraints.reinit(solution_relevant_partitioning);
		DoFTools::make_hanging_node_constraints(dof_handler, boundary_constraints);
		VectorTools::interpolate_boundary_values(dof_handler, 0, BoundaryValuesTemperatures<dim>(n_components), boundary_constraints);
		boundary_constraints.close();

		setup_matrices(solution_partitioning, solution_relevant_partitioning);

		system_rhs.reinit(solution_partitioning, mpi_communicator, true);

		newton_update.reinit(solution_partitioning, solution_relevant_partitioning, mpi_communicator);
		extension_vector.reinit(solution_partitioning, solution_relevant_partitioning, mpi_communicator);
		residual.reinit(solution_partitioning, solution_relevant_partitioning, mpi_communicator);
		evaluation_point.reinit(solution_partitioning, solution_relevant_partitioning, mpi_communicator);
		if(initial_step)
		{
			present_solution.reinit(solution_partitioning, solution_relevant_partitioning, mpi_communicator);
			old_solution.reinit(solution_partitioning, solution_relevant_partitioning, mpi_communicator);
		}
//		print_status_update(pcout, std::string("Setup done\n"), false);
	}

	template <int dim>
	void MinimalSurfaceProblem<dim>::assemble_system (const double Light_intensity, const double Light_intensity_old)
	{
		const QGauss<dim>  quadrature_formula(fe.degree+1);
		const FEValuesExtractors::Vector surface_N(Free_Carriers);
		const FEValuesExtractors::Vector surface_TE(Electron_Temperature);
		const FEValuesExtractors::Vector surface_TL(Lattice_Temperature);

		FEValues<dim> fe_values (fe, quadrature_formula,
								 update_values			  |
								 update_gradients         |
								 update_quadrature_points |
								 update_JxW_values);

		const unsigned int           dofs_per_cell = fe.dofs_per_cell;
		const unsigned int           n_q_points    = quadrature_formula.size();

		FullMatrix<double>           cell_matrix (dofs_per_cell, dofs_per_cell);
		Vector<double>               cell_rhs (dofs_per_cell);

		std::vector<types::global_dof_index>    local_dof_indices (dofs_per_cell);
		std::vector<double> residual_derivatives (dofs_per_cell);

		physics_equations<dim> local_equations(pulse_wavelength);

		for (auto cell = dof_handler.begin_active(); cell!=dof_handler.end(); ++cell)
		{
			if(cell->is_locally_owned())
			{
				cell_matrix = 0;
				cell_rhs = 0;

				fe_values.reinit (cell);

				cell->get_dof_indices(local_dof_indices);

				Table<3,Sacado::Fad::DFad<double> > grad_N (n_q_points, dim, dim), grad_N_old(n_q_points, dim, dim);
				Table<3,Sacado::Fad::DFad<double> > grad_TE (n_q_points, dim, dim), grad_TE_old(n_q_points, dim, dim);
				Table<3,Sacado::Fad::DFad<double> > grad_TL (n_q_points, dim, dim), grad_TL_old(n_q_points, dim, dim);
				Table<2, Sacado::Fad::DFad<double> > val_N (n_q_points, dim), val_N_old(n_q_points, dim);
				Table<2, Sacado::Fad::DFad<double> > val_TE (n_q_points, dim), val_TE_old(n_q_points, dim);
				Table<2, Sacado::Fad::DFad<double> > val_TL (n_q_points, dim), val_TL_old(n_q_points, dim);
				std::vector<Sacado::Fad::DFad<double> > ind_local_dof_values(dofs_per_cell);
				std::vector<double> local_dof_values(cell->get_fe().dofs_per_cell);
				cell->get_dof_values(present_solution, local_dof_values.begin(), local_dof_values.end());
				std::vector<Sacado::Fad::DFad<double>> local_dof_values_AD(cell->get_fe().dofs_per_cell);

				for (unsigned int i=0; i<dofs_per_cell; ++i)
				{
					ind_local_dof_values[i] = present_solution(local_dof_indices[i]);
					ind_local_dof_values[i].diff (i, dofs_per_cell);
				}

				for (unsigned int q=0; q<n_q_points; ++q)
					for (unsigned int d=0; d<dim; ++d)
						for(size_t local_dim = 0; local_dim < dim; ++local_dim)
						{
							grad_N[q][d][local_dim] = 0;
							grad_N_old[q][d][local_dim] = 0;
							grad_TE[q][d][local_dim] = 0;
							grad_TE_old[q][d][local_dim] = 0;
							grad_TL[q][d][local_dim] = 0;
							grad_TL_old[q][d][local_dim] = 0;
						}

				for (unsigned int q=0; q<n_q_points; ++q)
					for (unsigned int d=0; d<dim; ++d)
					{
						val_N[q][d] = 0;
						val_N_old[q][d] = 0;
						val_TE[q][d] = 0;
						val_TE_old[q][d] = 0;
						val_TL[q][d] = 0;
						val_TL_old[q][d] = 0;
					}

				for (unsigned int q=0; q<n_q_points; ++q)
				{
					for (unsigned int i=0; i<dofs_per_cell; ++i)
						for (unsigned int d = 0; d < dim; d++)
						{
							val_N[q][d] += ind_local_dof_values[i] * fe_values[surface_N].value(i, q)[d];
							val_N_old[q][d] += old_solution(local_dof_indices[i]) * fe_values[surface_N].value(i, q)[d];
							val_TE[q][d] += ind_local_dof_values[i] * fe_values[surface_TE].value(i, q)[d];
							val_TE_old[q][d] += old_solution(local_dof_indices[i]) * fe_values[surface_TE].value(i, q)[d];
							val_TL[q][d] += ind_local_dof_values[i] * fe_values[surface_TL].value(i, q)[d];
							val_TL_old[q][d] += old_solution(local_dof_indices[i]) * fe_values[surface_TL].value(i, q)[d];
							for(size_t local_dim = 0; local_dim < dim; ++local_dim)
							{
								grad_N[q][d][local_dim] += ind_local_dof_values[i] * fe_values[surface_N].gradient(i, q)[d][local_dim];
								grad_N_old[q][d][local_dim] += old_solution(local_dof_indices[i]) * fe_values[surface_N].gradient(i, q)[d][local_dim];
								grad_TE[q][d][local_dim] += ind_local_dof_values[i] * fe_values[surface_TE].gradient(i, q)[d][local_dim];
								grad_TE_old[q][d][local_dim] += old_solution(local_dof_indices[i]) * fe_values[surface_TE].gradient(i, q)[d][local_dim];
								grad_TL[q][d][local_dim] += ind_local_dof_values[i] * fe_values[surface_TL].gradient(i, q)[d][local_dim];
								grad_TL_old[q][d][local_dim] += old_solution(local_dof_indices[i]) * fe_values[surface_TL].gradient(i, q)[d][local_dim];
							}
						}
				}

				for (unsigned int i=0; i<dofs_per_cell; ++i)
				{
					Sacado::Fad::DFad<double> R_i = 0;

					for(unsigned int q=0; q<n_q_points; ++q)
					{
						for (unsigned int d = 0; d < dim; d++)
							for(size_t local_dim = 0; local_dim < dim; ++local_dim)
							{
								R_i += grad_TE[q][d][local_dim] * fe_values[surface_TE].gradient(i, q)[d][local_dim]/*(((val_TE[q][d] - val_TE_old[q][d])/time_step) * fe_values[surface_TE].value(i, q)[d]
										+ (theta * grad_TE[q][d][local_dim] + (1 - theta) * grad_TE_old[q][d][local_dim]) * fe_values[surface_TE].gradient(i, q)[d][local_dim])*/	* fe_values.JxW(q);
								R_i += (((val_N[q][d] - val_N_old[q][d])/time_step) * fe_values[surface_N].value(i, q)[d]
										+ (theta * grad_N[q][d][local_dim] + (1 - theta) * grad_N_old[q][d][local_dim]) * fe_values[surface_N].gradient(i, q)[d][local_dim])	* fe_values.JxW(q);
								R_i += (((val_TL[q][d] - val_TL_old[q][d])/time_step) * fe_values[surface_TL].value(i, q)[d]
										+ (theta * grad_TL[q][d][local_dim] + (1 - theta) * grad_TL_old[q][d][local_dim]) * fe_values[surface_TL].gradient(i, q)[d][local_dim])	* fe_values.JxW(q);
							}
					}

					for (unsigned int j=0; j<dofs_per_cell; ++j)
						residual_derivatives[j] = R_i.fastAccessDx(j);

					for (unsigned int j=0; j<dofs_per_cell; ++j)
						cell_matrix(i, j) += residual_derivatives[j];

					cell_rhs(i) -= R_i.val();
				}

				cell->get_dof_indices (local_dof_indices);

				newton_constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
			}
		}

		system_matrix.compress(VectorOperation::add);
		system_rhs.compress(VectorOperation::add);

		print_status_update(pcout, std::string("Returning from assemble_matrix\n"), false);

	}



	// @sect4{MinimalSurfaceProblem::solve}


	template <int dim>
	void MinimalSurfaceProblem<dim>::solve (const double I_val, const double I_val_old)
	{

		IndexSet solution_relevant_partitioning(dof_handler.n_dofs());
		DoFTools::extract_locally_relevant_dofs(dof_handler, solution_relevant_partitioning);
		LinearAlgebraTrilinos::MPI::Vector completely_distributed_solution(dof_handler.locally_owned_dofs(), mpi_communicator);
		LinearAlgebraTrilinos::MPI::Vector completely_distributed_update(dof_handler.locally_owned_dofs(), mpi_communicator);

		completely_distributed_solution = present_solution;

		SolverControl solver_control (dof_handler.n_dofs(),
									  (system_rhs.l2_norm() > 0) ? 1e-8 * system_rhs.l2_norm() : 1e-8);
		LinearAlgebraTrilinos::SolverGMRES  solver (solver_control);

		LinearAlgebraTrilinos::MPI::PreconditionAMG preconditioner;
		LinearAlgebraTrilinos::MPI::PreconditionAMG::AdditionalData data;

		print_status_update(pcout, "Initializing system matrix with preconditioner\n");
		preconditioner.initialize(system_matrix, data);
		print_status_update(pcout, "Solving\n");
		solver.solve (system_matrix, completely_distributed_update, system_rhs,
					  preconditioner);
		print_status_update(pcout, "Solving done\n");

		hanging_node_constraints.distribute (completely_distributed_update);

		print_status_update(pcout, "Adding to newton\n");

		newton_update = completely_distributed_update;
		print_status_update(pcout, std::string("L2-norm of newton-update: ") + std::to_string(completely_distributed_update.l2_norm()*1e8) + std::string("\n"), true);

		newton_update.compress(VectorOperation::insert);

		const double alpha = recalculate_step_length(I_val, I_val_old);
		completely_distributed_solution.add (alpha, completely_distributed_update);

		present_solution = completely_distributed_solution;
		print_status_update(pcout, std::string("Returning from solve()\n"), false);

	}

	// @sect4{MinimalSurfaceProblem::refine_mesh}

	// The first part of this function is the same as in step-6... However,
	// after refining the mesh we have to transfer the old solution to the new
	// one which we do with the help of the SolutionTransfer class. The process
	// is slightly convoluted, so let us describe it in detail:
	template <int dim>
	void MinimalSurfaceProblem<dim>::refine_mesh ()
	{

		print_status_update(pcout, "Refining mesh\n", false);
		Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

		KellyErrorEstimator<dim>::estimate (dof_handler,
											QGauss<dim-1>(fe.degree+1),
											typename FunctionMap<dim>::type(),
											present_solution,
											estimated_error_per_cell,
											ComponentMask(),
											nullptr,
											0,
											triangulation.locally_owned_subdomain());

		parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number (triangulation,
																				estimated_error_per_cell,
																				0.3, 0.03);

		if (triangulation.n_levels() > max_grid_level)
			for (auto cell = triangulation.begin_active(max_grid_level); cell != triangulation.end(); ++cell)
				cell->clear_refine_flag ();
		for (auto cell = triangulation.begin_active(min_grid_level); cell != triangulation.end_active(min_grid_level); ++cell)
			cell->clear_coarsen_flag ();

		print_status_update(pcout, "Preparing refinement\n");
		triangulation.prepare_coarsening_and_refinement ();

		parallel::distributed::SolutionTransfer<dim, LinearAlgebraTrilinos::MPI::Vector> solution_transfer(dof_handler), old_solution_transfer(dof_handler);
		solution_transfer.prepare_for_coarsening_and_refinement(present_solution);
		old_solution_transfer.prepare_for_coarsening_and_refinement(old_solution);
		triangulation.execute_coarsening_and_refinement();

		setup_system(true);
		print_status_update(pcout, std::string("Interpolation\n"), false);
		LinearAlgebraTrilinos::MPI::Vector distributed_solution(dof_handler.locally_owned_dofs(), mpi_communicator);
		LinearAlgebraTrilinos::MPI::Vector distributed_old_solution(dof_handler.locally_owned_dofs(), mpi_communicator);
		print_status_update(pcout, std::string("Doing interpolation\n"), false);
		solution_transfer.interpolate(distributed_solution);
		old_solution_transfer.interpolate(distributed_old_solution);

		print_status_update(pcout, std::string("Adding boundary conditions\n"), false);
		boundary_constraints.distribute(distributed_solution);
		boundary_constraints.distribute(distributed_old_solution);
		present_solution = distributed_solution;
		old_solution = distributed_old_solution;

		setup_system(false);

		print_status_update(pcout, "Almost done with interpolation\n");
		//setup_system (false);

	}



	// @sect4{MinimalSurfaceProblem::set_boundary_values}

	// The next function ensures that the solution vector's entries respect the
	// boundary values for our problem.  Having refined the mesh (or just
	// started computations), there might be new nodal points on the
	// boundary. These have values that are simply interpolated from the
	// previous mesh (or are just zero), instead of the correct boundary
	// values. This is fixed up by setting all boundary nodes explicit to the
	// right value:
	template <int dim>
	void MinimalSurfaceProblem<dim>::set_boundary_values ()
	{
		boundary_constraints.distribute(present_solution);
		boundary_constraints.distribute(old_solution);
		print_status_update(pcout, "Closed\n");
	}


	// @sect4{MinimalSurfaceProblem::compute_residual}

	template <int dim>
	double MinimalSurfaceProblem<dim>::compute_residual (const double alpha, const double Light_intensity, const double Light_intensity_old)
	{
		const FEValuesExtractors::Vector surface_N(Free_Carriers);
		const FEValuesExtractors::Vector surface_TE(Electron_Temperature);
		const FEValuesExtractors::Vector surface_TL(Lattice_Temperature);
		IndexSet solution_relevant_partitioning(dof_handler.n_dofs());
		DoFTools::extract_locally_relevant_dofs(dof_handler, solution_relevant_partitioning);
		//LinearAlgebraTrilinos::MPI::Vector local_evaluation_point(dof_handler.locally_owned_dofs(), mpi_communicator), local_extension_vector(dof_handler.locally_owned_dofs(), mpi_communicator);

		evaluation_point = present_solution;
		extension_vector = newton_update;
		extension_vector *= alpha;
		evaluation_point += extension_vector;

		LinearAlgebraTrilinos::MPI::Vector local_residual(dof_handler.locally_owned_dofs(), mpi_communicator);
		//evaluation_point.add (alpha, newton_update);
		//evaluation_point.compress(VectorOperation::add);
		print_status_update(pcout, "Creating other stuff\n");

		const QGauss<dim>  quadrature_formula(fe.degree+1);
		FEValues<dim> fe_values (fe, quadrature_formula,
								 update_values			  |
								 update_gradients         |
								 update_quadrature_points |
								 update_JxW_values);

		const unsigned int           dofs_per_cell = fe.dofs_per_cell;
		const unsigned int           n_q_points    = quadrature_formula.size();

		Vector<double>               cell_residual (dofs_per_cell);
		std::vector<Tensor<2, dim> > gradients_N(n_q_points), old_gradients_N(n_q_points);
		std::vector<Tensor<1, dim> > values_N(n_q_points), old_values_N(n_q_points);
		std::vector<Tensor<2, dim> > gradients_TE(n_q_points), old_gradients_TE(n_q_points);
		std::vector<Tensor<1, dim> > values_TE(n_q_points), old_values_TE(n_q_points);
		std::vector<Tensor<2, dim> > gradients_TL(n_q_points), old_gradients_TL(n_q_points);
		std::vector<Tensor<1, dim> > values_TL(n_q_points), old_values_TL(n_q_points);

		std::vector<types::global_dof_index>    local_dof_indices (dofs_per_cell);

		physics_equations<dim> local_equations(pulse_wavelength);

		print_status_update(pcout, std::string("Starting looping over cells in residual\n"), false);
		for (auto cell = dof_handler.begin_active(); cell!=dof_handler.end(); ++cell)
		{
			if(cell->is_locally_owned())
			{
				cell_residual = 0;
				fe_values.reinit (cell);

				fe_values[surface_N].get_function_values(evaluation_point, values_N);
				fe_values[surface_N].get_function_values(old_solution, old_values_N);
				fe_values[surface_N].get_function_gradients (evaluation_point,
															 gradients_N);
				fe_values[surface_N].get_function_gradients (old_solution,
															 old_gradients_N);

				fe_values[surface_TE].get_function_values(evaluation_point, values_TE);
				fe_values[surface_TE].get_function_values(old_solution, old_values_TE);
				fe_values[surface_TE].get_function_gradients (evaluation_point,
															  gradients_TE);
				fe_values[surface_TE].get_function_gradients (old_solution,
															  old_gradients_TE);

				fe_values[surface_TL].get_function_values(evaluation_point, values_TL);
				fe_values[surface_TL].get_function_values(old_solution, old_values_TL);
				fe_values[surface_TL].get_function_gradients (evaluation_point,
															  gradients_TL);
				fe_values[surface_TL].get_function_gradients (old_solution,
															  old_gradients_TL);


				print_status_update(pcout, std::string("Looping over points\n"), false);
				for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
				{
					print_status_update(pcout, std::string("Updating residual side\n"), false);
					for (unsigned int i = 0; i < dofs_per_cell; ++i)
					{
						for(size_t d = 0; d < dim; ++d)
						{
							for(size_t local_dim = 0; local_dim < dim; ++local_dim)
							{
								cell_residual(i) -= gradients_TE[q_point][d][local_dim] * fe_values[surface_TE].gradient(i, q_point)[d][local_dim]/*(((values_TE[q_point][d] - old_values_TE[q_point][d])/time_step) * fe_values[surface_TE].value(i, q_point)[d]
													 + (theta * gradients_TE[q_point][d][local_dim] + (1 - theta) * old_gradients_TE[q_point][d][local_dim]) * fe_values[surface_TE].gradient(i, q_point)[d][local_dim])*/
										* fe_values.JxW(q_point);
								cell_residual(i) -= (((values_TL[q_point][d] - old_values_TL[q_point][d])/time_step) * fe_values[surface_TL].value(i, q_point)[d]
													 + (theta * gradients_TL[q_point][d][local_dim] + (1 - theta) * old_gradients_TL[q_point][d][local_dim]) * fe_values[surface_TL].gradient(i, q_point)[d][local_dim])
										* fe_values.JxW(q_point);
								cell_residual(i) -= (((values_N[q_point][d] - old_values_N[q_point][d])/time_step) * fe_values[surface_N].value(i, q_point)[d]
													 + (theta * gradients_N[q_point][d][local_dim] + (1 - theta) * old_gradients_N[q_point][d][local_dim]) * fe_values[surface_N].gradient(i, q_point)[d][local_dim])
										* fe_values.JxW(q_point);
							}
						}
					}
				}

				cell->get_dof_indices (local_dof_indices);
				print_status_update(pcout, std::string("Distributing values\n"), false);
				boundary_constraints.distribute_local_to_global(cell_residual, local_dof_indices, local_residual);
			}
		}
		print_status_update(pcout, std::string("Done looping\n"), false);
		local_residual.compress(VectorOperation::add);
		boundary_constraints.set_zero(local_residual);
		return local_residual.l2_norm();

	}

	template <int dim>
	double MinimalSurfaceProblem<dim>::determine_step_length()
	{
		return alpha_val;
	}

	template <int dim>
	double MinimalSurfaceProblem<dim>::recalculate_step_length(const double I_val, const double I_val_old)
	{
		std::vector<double> potential_step_sizes = {0.1, 0.25}, potential_residual_values;
		for(size_t i = 0; i < potential_step_sizes.size(); ++i)
		{
			auto residual = compute_residual(potential_step_sizes[i], I_val, I_val_old);
			potential_residual_values.push_back(residual);
			//print_status_update(pcout, std::string("Residual value for ") + std::to_string(potential_step_sizes[i]) + std::string(" is ") + std::to_string(compute_residual(potential_step_sizes[i], I_val, I_val_old)) + std::string("\n"), true);
		}

		size_t lowest_residual = std::min_element(potential_residual_values.begin(), potential_residual_values.end()) - potential_residual_values.begin();
		//print_status_update(pcout, std::string("First residual value ") + std::to_string(*potential_step_sizes.begin()) + std::string(" and last residual value ") + std::to_string(*potential_residual_values.end()) + std::string("\n"), true);
		//print_status_update(pcout, std::string("Difference between the first and last element ") + std::to_string(*potential_residual_values.begin() - *potential_residual_values.end()) + std::string("\n"), true);
		//		if(*potential_residual_values.end() == potential_residual_values[lowest_residual])
		//			alpha_val = *potential_step_sizes.end();
		//		else
		return potential_step_sizes[lowest_residual];
		//print_status_update(pcout, std::string("Choosing ") + std::to_string(alpha_val) + std::string(" as new alpha_val\n"), true);
		//		size_t highest_residual = std::max_element(potential_residual_values.begin(), potential_residual_values.end()) - potential_residual_values.begin();
		//		print_status_update(pcout, std::string("Lowest residual is for ") + std::to_string(potential_step_sizes[lowest_residual]) + std::string(" with ") + std::to_string(*(std::min_element(potential_residual_values.begin(), potential_residual_values.end()))) + std::string("\n"), true);
		//		print_status_update(pcout, std::string("Highest residual is for ") + std::to_string(potential_step_sizes[highest_residual]) + std::string(" with ") + std::to_string(*(std::max_element(potential_residual_values.begin(), potential_residual_values.end()))) + std::string("\n"), true);

	}


	template <int dim>
	void MinimalSurfaceProblem<dim>::output_results (const int cycle) const
	{
		print_status_update(pcout, "Writing data to disk\n");
		DataOut<dim> data_out;
		data_out.attach_dof_handler (dof_handler);
		std::vector<std::string> solution_names = std::vector<std::string>{"solution_A", "solution_A", "solution_B", "solution_B", "solution_C", "solution_C"};
		std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(n_components * dim, DataComponentInterpretation::component_is_part_of_vector);
		data_out.add_data_vector (newton_update, solution_names, DataOut<dim>::type_dof_data, data_component_interpretation);
		Vector<float> subdomain (triangulation.n_active_cells());
		for (unsigned int i=0; i<subdomain.size(); ++i)
			subdomain(i) = triangulation.locally_owned_subdomain();
		data_out.add_data_vector (subdomain, "subdomain");
		data_out.build_patches ();
		const std::string filename = ("solution-" +
									  Utilities::int_to_string (cycle, 2) +
									  "." +
									  Utilities::int_to_string
									  (triangulation.locally_owned_subdomain(), 4));
		std::ofstream output ((filename + ".vtu").c_str());
		data_out.write_vtu (output);
		if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
		{
			std::vector<std::string> filenames;
			for (unsigned int i=0;
				 i<Utilities::MPI::n_mpi_processes(mpi_communicator);
				 ++i)
				filenames.push_back ("solution-" +
									 Utilities::int_to_string (cycle, 2) +
									 "." +
									 Utilities::int_to_string (i, 4) +
									 ".vtu");
			std::ofstream master_output (("solution-" +
										  Utilities::int_to_string (cycle, 2) +
										  ".pvtu").c_str());
			data_out.write_pvtu_record (master_output, filenames);
		}
		print_status_update(pcout, "Finished writing data\n", false);
	}


	// @sect4{MinimalSurfaceProblem::run}

	// In the run function, we build the first grid and then have the top-level
	// logic for the Newton iteration. The function has two variables, one that
	// indicates whether this is the first time we solve for a Newton update and
	// one that indicates the refinement level of the mesh

	template <int dim>
	void MinimalSurfaceProblem<dim>::run ()
	{
		unsigned int refinement = 0;
		bool         first_step = true;

		// As described in the introduction, the domain is the unit disk around
		// the origin, created in the same way as shown in step-6. The mesh is
		// globally refined twice followed later on by several adaptive cycles:
		GridGenerator::hyper_ball (triangulation);
		static const SphericalManifold<dim> boundary;
		//GridGenerator::hyper_cube(triangulation, -1., 1.);
		//		static const StraightBoundary<dim> boundary(dim);
		triangulation.set_all_manifold_ids_on_boundary(0);
		triangulation.set_manifold (0, boundary);
		//triangulation.set_boundary(1);
//		for(auto cell = triangulation.begin(); cell != triangulation.end(); ++cell)
//			for(size_t face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
//				//if((cell->face(face_number)->center()(0) < PENETRATION_DEPTH))
//				cell->face(face_number)->set_boundary_id(0);
//		for(auto cell = triangulation.begin(); cell != triangulation.end(); ++cell)
//			for(size_t face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
//				if((cell->face(face_number)->center()(0) < -0.99) || (cell->face(face_number)->center()(0) > 0.99))
//					cell->face(face_number)->set_boundary_id(1);
		triangulation.refine_global(min_grid_level);

		double previous_res = 0;
		double cur_time = start_time;

		const double cur0_time = cur_time;
		double I_val = 0, I_val_old = 0;
		I_val = pulse_intensity * calculate_gaussian_pulse(cur_time, pulse_duration);
		I_val_old = pulse_intensity * calculate_gaussian_pulse(cur_time - time_step, pulse_duration);
		double I_peak = pulse_intensity * calculate_gaussian_pulse(0, pulse_duration);
		print_status_update(pcout, std::string("Peak light intensity is ") + std::to_string(I_peak) + std::string("\n"), true);
		print_status_update(pcout, std::string("Starting light intensity is ") + std::to_string(I_val) + std::string(" (") + std::to_string(I_peak == 0?0:I_val/I_peak*100) + std::string("%)\n"), true);
		physics_equations<dim> local_equations(pulse_wavelength);
		//bool refine_this_mesh = false;
		bool next_step = false;
		std::vector<double> old_residuals;
		const double previous_res_threshold = 1e-3;

		while(cur_time < max_time)
		{
			while (first_step || ((previous_res>previous_res_threshold) && !next_step))
			{
				if (first_step == true)
				{
					print_status_update(pcout, std::string("******** Initial mesh ********\n"), true);
					print_status_update(pcout, "Setting up system\n");
					setup_system (true);
					print_status_update(pcout, "Setting boundary values\n");

					IndexSet solution_relevant_partitioning;
					LinearAlgebraTrilinos::MPI::Vector local_solution;
					local_solution.reinit(dof_handler.locally_owned_dofs(), mpi_communicator);
					DoFTools::extract_locally_relevant_dofs(dof_handler, solution_relevant_partitioning);
					print_status_update(pcout, std::string("Before distribution\n"), true);
//					VectorTools::project (dof_handler,
//										  hanging_node_constraints,
//										  QGauss<dim>(fe.degree+1),
//										  InitialValues<dim>(),
//										  local_solution);
					VectorTools::interpolate(dof_handler, InitialValues<dim>(), local_solution);

					boundary_constraints.distribute(local_solution);
					old_solution = local_solution;
					present_solution = local_solution;
					//std::cout << "After distribution\n";
					print_status_update(pcout, std::string("After distribution\n"), true);
					first_step = false;
					output_results(0);

				}

				else
					if(true)
					{
						++refinement;
						print_status_update(pcout, std::string("******** Refined mesh ") + std::to_string(refinement) + std::string(" ********\n"), true);

						refine_mesh();
						output_results(100);
					}

				print_status_update(pcout, std::string("Calculating initial residual\nInitial Residual: ") + std::to_string(compute_residual(0, I_val, I_val_old)) + std::string("\n"), true);

				for (size_t inner_iteration = 0; inner_iteration < max_inner_iterations; ++inner_iteration)
				{
					print_status_update(pcout, "Assembling system\n");
					setup_system(false);
					assemble_system (I_val, I_val_old);
					print_status_update(pcout, "Calculating previous res\n");
					previous_res = system_rhs.l2_norm();
					print_status_update(pcout, std::string("Current l2 norm is " + std::to_string(previous_res)), true);
					print_status_update(pcout, "Trying to solve\n");
					solve (I_val, I_val_old);
					output_results(inner_iteration+1);
					//recalculate_step_length(I_val, I_val_old);
					print_status_update(pcout, std::string("  Residual: ") + std::to_string(compute_residual(0, I_val, I_val_old)) + std::string("\n"), true);
					//refine_this_mesh = true;
				}
				return;
				if(refinement > 1000)
					next_step = true;
				if(previous_res > previous_res_threshold)
				{
					if(old_residuals.size() == 0)
					{
						print_status_update(pcout, std::string("Added initial value ") + std::to_string(previous_res) + std::string(" to vector\n"), true);
						old_residuals.push_back(previous_res);
					}
					else
					{
						if(abs(previous_res - old_residuals[old_residuals.size() - 1]) < 1e-8)
						{
							if(old_residuals.size() < max_same_res_values)
							{
								print_status_update(pcout, std::string("Adding new value ") + std::to_string(previous_res) + std::string(" to old_residuals with now a size of ") + std::to_string(old_residuals.size()) + std::string("\n"), true);
								old_residuals.push_back(previous_res);
							}
							else
							{
								//                            print_status_update(pcout, std::string("Doing next step!\n"), false);
								//							double res_0 = compute_residual(0, I_val, I_val_old);
								//							double res_001 = compute_residual(0.01, I_val, I_val_old);
								//							double res_01 = compute_residual(0.1, I_val, I_val_old);
								//							double res_1 = compute_residual(1, I_val, I_val_old);
								//							print_status_update(pcout, std::string("res_0: ") + std::to_string(res_0) + std::string(", res_001: ") + std::to_string(res_001) + std::string(", res_01: ") + std::to_string(res_01) + std::string(", res_1: ") + std::to_string(res_1) + std::string("\n"), true);
								//							getchar();
								old_residuals.clear();
								next_step = true;
							}
						}
						else
						{
							print_status_update(pcout, std::string("New residual is ") + std::to_string((previous_res - old_residuals[old_residuals.size() - 1])*1e9) + std::string("\n"), false);
							print_status_update(pcout, std::string("Last residual is ") + std::to_string(old_residuals[old_residuals.size() - 1]*1e9) + std::string("\n"), false);
							print_status_update(pcout, std::string("Current residual is ") + std::to_string((previous_res)*1e9) + std::string("\n"), false);
							old_residuals.clear();
							old_residuals.push_back(previous_res);
						}
					}
				}
			}

			next_step = false;
			alpha_val = 0.1;
			old_residuals.clear();
			//refine_this_mesh = false;
			print_status_update(pcout, std::string("*********Time step to : ") + std::to_string(cur_time/time_step) + std::string(" of ") + std::to_string(max_time/time_step) + std::string(" Steps *********\n"), true);
			cur_time += time_step;
			old_solution = present_solution;
			output_results(cur_time/time_step + abs(cur0_time)/time_step);
			old_solution.compress(VectorOperation::insert);
			//output_results(cur_time/time_step + abs(cur0_time)/time_step);
			I_val = pulse_intensity * calculate_gaussian_pulse(cur_time, pulse_duration);
			I_val_old = pulse_intensity * calculate_gaussian_pulse(cur_time - time_step, pulse_duration);
			print_status_update(pcout, std::string("New light intensity is ") + std::to_string(I_val) + std::string(" (") + std::to_string(I_val/I_peak*100) + std::string("%)\n"), true);
			print_status_update(pcout, std::string("New heat source value is ") + std::to_string(calculate_heat_source_TE(pulse_wavelength, I_val, local_equations.ambient_temperature, local_equations.initial_carrier_density, dim)) + std::string("\n"), true);
			previous_res = 10;
			refinement = 0;
			print_status_update(pcout, std::string("Size of newton_matrix: ") + std::to_string(newton_constraints.memory_consumption()) + std::string("\n"), true);
			print_status_update(pcout, std::string("Size of boundary_matrix: ") + std::to_string(boundary_constraints.memory_consumption()) + std::string("\n"), true);
			print_status_update(pcout, std::string("Size of hanging_node_matrix: ") + std::to_string(hanging_node_constraints.memory_consumption()) + std::string("\n"), true);
			print_status_update(pcout, std::string("Amount of unknowns: ") + std::to_string(dof_handler.n_dofs()) + std::string("\n"), true);
		}
	}

//	double one_photon_absorption(const double lattice_temperature, const double wavelength)
//	{
//		physics_equations<2> local_equations(wavelength);
//		const double photon_energy = local_equations.photon_energy_func() / local_equations.electron_charge;
//		//const double alpha_1 = 0.504 * sqrt(photon_energy) + 0.068 * pow(photon_energy - 0.0055, 2);
//		const double theta_1 = 212;
//		//const double alpha_2 = 18.08 * sqrt(photon_energy) + 1 * pow(photon_energy - 0.0055, 2);
//		const double theta_2 = 670;
//		//const double alpha[] = {alpha_1, alpha_2};
//		const double theta[] = {theta_1, theta_2};
//		double bandgap_energy = local_equations.E_g(lattice_temperature) / local_equations.electron_charge;
//		//std::cout << "Alpha_1 is " << alpha_1 << " and alpha_2 is " << alpha_2 << '\n';
//		std::cout << "Bandgap energy is " << bandgap_energy << '\n';
//		double absorption_coefficient = 0;

//		for(size_t i = 1; i <= 2; ++i)
//			for(size_t l = 1; l <= 2; ++l)
//			{
//				double alpha_val = 0;
//				if(i == 1)
//				{
//					std::cout << "Internal energy is " << photon_energy - bandgap_energy + pow(-1, l) * theta[i-1] << '\n';
//					alpha_val = alpha_1(photon_energy - bandgap_energy + pow(-1, l) * theta[i-1]);
//				}
//				else
//				{
//					std::cout << "Internal energy is " << photon_energy - bandgap_energy + pow(-1, l) * theta[i-1] << '\n';
//					alpha_val = alpha_2(photon_energy - bandgap_energy + pow(-1, l) * theta[i-1]);
//				}
//				//std::cout << "Absorption coefficient (" << i << ", " << l << ") is " << pow(-1, l) * (alpha[i-1] * (photon_energy - bandgap_energy + pow(-1, l) * theta[i-1]))/(exp(pow(-1, l) * theta[i-1]/lattice_temperature) - 1) << '\n';
//				std::cout << "Absorption coefficient (" << i << ", " << l << ") is " << pow(-1, l) * (alpha_val)/(exp(pow(-1, l) * theta[i-1]/lattice_temperature) - 1) << '\n';
//			}
//	}

}

// @sect4{The main function}



// Finally the main function. This follows the scheme of all other main
// functions:
int main (int argc, char *argv[])
{
	//    Step15::physics_equations<1> local_equations;
	//    std::cout << local_equations.mu_e(local_equations.ambient_temperature) << '\t' << local_equations.mu_h(local_equations.ambient_temperature) << '\n' << local_equations.k_E(local_equations.initial_carrier_density, local_equations.ambient_temperature, local_equations.ambient_temperature) << '\n';
	//    return 0;
	try
	{
		using namespace dealii;
		using namespace Step15;

		Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
//		physics_equations<1> local_equations(1152e-9);
//		std::cout << "k_N: " << local_equations.k_N(293.) << '\n';
//		std::cout << "k_L: " << local_equations.k_L(293.) << '\n';
//		std::cout << "k_E: " << local_equations.k_E(local_equations.initial_carrier_density, 293., 293.) << '\n';
//		std::cout << "Photon energy: " << local_equations.photon_energy_func() / local_equations.electron_charge << '\n';
//		return 0;

		char * file_name = argv[1];
		if(argc < 2)
			file_name = (char*)(std::string("parameters.prm").c_str());
		MinimalSurfaceProblem<2> laplace_problem_2d(file_name);
		laplace_problem_2d.run ();
	}
	catch (std::exception &exc)
	{
		std::cerr << std::endl << std::endl
				  << "----------------------------------------------------"
				  << std::endl;
		std::cerr << "Exception on processing: " << std::endl
				  << exc.what() << std::endl
				  << "Aborting!" << std::endl
				  << "----------------------------------------------------"
				  << std::endl;

		return 1;
	}
	catch (...)
	{
		std::cerr << std::endl << std::endl
				  << "----------------------------------------------------"
				  << std::endl;
		std::cerr << "Unknown exception!" << std::endl
				  << "Aborting!" << std::endl
				  << "----------------------------------------------------"
				  << std::endl;
		return 1;
	}
	return 0;
}
